"""
MagVLA Policy (Version 13 - Flow Matching 版本，基于 Qwen2.5-VL-7B)

关键修改（相对 version13-IX-diffusion）：
- DiffusionChunkHead → FlowMatchingChunkHead（conditional flow matching / rectified flow）
- 不再依赖 diffusers（flow matching 无需 noise scheduler）
- chunk_loss 改为 velocity prediction MSE（与 diffusion 的 noise MSE 在数值上可比）
- 推理时跑 Euler 10步积分生成动作

Flow Matching 原理（π₀ 使用的技术，Lipman et al. 2023）：
1. 训练：
   - 采样 z ~ N(0, I) 和 t ~ U(0, 1)
   - 构造直线路径：x_t = (1-t) * z + t * action_gt
   - 目标速度：v_target = action_gt - z（直线路径上的常速度）
   - 模型预测速度：v_pred = model(x_t, cond, t)
   - Loss: MSE(v_pred, v_target)
2. 推理：
   - 从 z ~ N(0, I) 开始
   - Euler 积分：x_{t+dt} = x_t + dt * model(x_t, cond, t)
   - 10 步后输出作为动作
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple
import config

__all__ = ["MagVLAPolicy", "build_policy"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase Head
# ─────────────────────────────────────────────────────────────────────────────
class PhaseHead(nn.Module):
    """
    输入: vlm_hidden [B, L, D], token_pad_mask [B, L] (True=padding, False=valid)
    输出:
        phase_logits   [B, 2]          # 分类 logits (用于 L_phase) — 2 类：approach vs transport
        phase_token    [B, D]          # 可学习的 phase embedding (用于条件 ACT)
        phase_id       [B]             # argmax(phase_logits) 推理时用
    """

    def __init__(self, hidden_dim: int = 3584, state_dim: int = 4, num_phases: int = 2):
        super().__init__()
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim

        # 状态特征投影（当前位置）
        self.state_proj_phase = nn.Linear(state_dim, 64)

        # 运动方向投影（最新帧 - 最旧帧的状态差）
        self.motion_proj = nn.Linear(state_dim, 64)

        # 分类器：vision_pool(D) + state(64) + motion(64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_phases),
        )

        # 可学习的 phase embedding（给 ACT head 用）
        self.phase_embedding = nn.Embedding(num_phases, hidden_dim)
        nn.init.normal_(self.phase_embedding.weight, std=0.02)

    def forward(
        self,
        vlm_hidden: torch.Tensor,
        phase_gt: Optional[torch.Tensor] = None,
        token_pad_mask: Optional[torch.Tensor] = None,
        state_normalized: Optional[torch.Tensor] = None,
        state_history: Optional[torch.Tensor] = None,
    ):
        B, L, D = vlm_hidden.shape
        dtype = vlm_hidden.dtype

        # ── 1. Vision pool（masked mean pooling，同之前）──────────────
        if token_pad_mask is not None:
            valid_mask = (~token_pad_mask).unsqueeze(-1).to(dtype)
            valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
            vision_pool = (vlm_hidden * valid_mask).sum(dim=1) / valid_count
        else:
            vision_pool = vlm_hidden.mean(dim=1)

        # ── 2. State 特征（当前位置）─────────────────────────────────
        if state_normalized is not None:
            state_feat = self.state_proj_phase(state_normalized.to(dtype))
        else:
            state_feat = torch.zeros(B, 64, device=vlm_hidden.device, dtype=dtype)

        # ── 3. Motion 特征（运动方向 = 最新帧 - 最旧帧）─────────────
        if state_history is not None:
            motion = state_history[:, -1, :] - state_history[:, 0, :]
            motion_feat = self.motion_proj(motion.to(dtype))
        else:
            motion_feat = torch.zeros(B, 64, device=vlm_hidden.device, dtype=dtype)

        # ── 4. 拼接 → 分类 ──────────────────────────────────────────
        combined = torch.cat([vision_pool, state_feat, motion_feat], dim=-1)
        phase_logits = self.classifier(combined)

        # ── 5. Teacher forcing（训练）/ Argmax（推理）──────────────
        if self.training and phase_gt is not None:
            phase_ids = phase_gt
        else:
            phase_ids = phase_logits.argmax(dim=-1)
        phase_token = self.phase_embedding(phase_ids)

        return {
            "phase_logits": phase_logits,
            "phase_token": phase_token,
            "phase_id": phase_ids,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Flow Matching Chunk Head（conditional flow matching / rectified flow 小脑）
# ─────────────────────────────────────────────────────────────────────────────
class FlowMatchingChunkHead(nn.Module):
    """
    输入:
        vlm_hidden: [B, L, D]       # VLM backbone 输出
        phase_token: [B, D]          # Phase Head 输出的条件向量
        memory_pad_mask: [B, L]      # BoolTensor, True=padding
        action_gt: [B, K, A] or None
                                   # 训练时传入 → 计算 velocity 预测 MSE 损失
                                   # 推理时传 None → 跑 Euler 积分

    输出 (dict):
        chunk_loss: scalar tensor or None
        action_chunk: [B, K, A]    # 训练时 = action_gt（passthrough）
                                   # 推理时 = Euler 积分结果

    设计说明：
    - 用 condition vector（pooled VLM hidden + phase_token）作为 velocity network 的 global condition
    - Velocity network 与 diffusion 的 denoiser 结构完全相同（简单 MLP），便于公平对比
    - 训练：
        采样 z ~ N(0,I), t ~ U(0,1)
        x_t = (1-t) * z + t * action_gt       ← 直线路径
        v_target = action_gt - z              ← 沿直线的常速度
        v_pred = MLP(x_t, cond, t_embed)
        loss = MSE(v_pred, v_target)
    - 推理：
        x = z ~ N(0,I)
        dt = 1/num_inference_steps
        for i in range(num_inference_steps):
            t = i * dt
            v = MLP(x, cond, t_embed)
            x = x + dt * v
        return x
    - dtype 处理：路径构造用 fp32（数值稳定），MLP 跟随 vlm_hidden 的 dtype（bf16）
    - 时间 embedding：t ∈ [0,1] 连续，通过 TIME_SCALE 放大到 ~1000 级别再过 sinusoidal
      （避免所有 t 落在低频部分导致 embedding 无法区分）
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        action_dim: int = 4,
        chunk_size: int = 5,
        cond_dim: int = 256,
        time_dim: int = 128,
        num_inference_steps: int = 10,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.num_inference_steps = num_inference_steps
        self.time_scale = time_scale

        action_flat = chunk_size * action_dim  # 5 * 4 = 20

        # 条件投影：vlm_hidden 池化 + phase_token
        self.vlm_proj = nn.Linear(hidden_dim, cond_dim)
        self.phase_proj = nn.Linear(hidden_dim, cond_dim)

        # 时间 embedding（sinusoidal → MLP）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Velocity network MLP（输入：x_t + cond + time_emb）
        # 结构与 diffusion denoiser 完全相同，便于公平对比
        net_in = action_flat + cond_dim * 2 + time_dim
        net_hidden = 512
        self.velocity_net = nn.Sequential(
            nn.Linear(net_in, net_hidden),
            nn.SiLU(),
            nn.Linear(net_hidden, net_hidden),
            nn.SiLU(),
            nn.Linear(net_hidden, net_hidden),
            nn.SiLU(),
            nn.Linear(net_hidden, action_flat),
        )
        # 输出层初始化为 0，让初始速度接近 0（稳定的训练起点）
        nn.init.zeros_(self.velocity_net[-1].weight)
        nn.init.zeros_(self.velocity_net[-1].bias)

    def _sinusoidal_time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding for continuous timesteps t ∈ [0, 1].
        t: [B] float
        returns: [B, time_dim] fp32
        """
        device = t.device
        # Scale t ∈ [0,1] → [0, time_scale] 避免低频部分退化
        t_scaled = t.float() * self.time_scale

        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, dtype=torch.float32, device=device)
            / half
        )
        args = t_scaled.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, time_dim]
        return emb

    def _get_cond(
        self,
        vlm_hidden: torch.Tensor,
        phase_token: torch.Tensor,
        memory_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        构造条件向量：masked mean pool(vlm_hidden) + phase_token → [B, 2*cond_dim]
        """
        dtype = vlm_hidden.dtype
        if memory_pad_mask is not None:
            valid = (~memory_pad_mask).unsqueeze(-1).to(dtype)
            valid_count = valid.sum(dim=1).clamp_min(1.0)
            pooled = (vlm_hidden * valid).sum(dim=1) / valid_count
        else:
            pooled = vlm_hidden.mean(dim=1)

        vlm_cond = self.vlm_proj(pooled)             # [B, cond_dim]
        phase_cond = self.phase_proj(phase_token)     # [B, cond_dim]
        cond = torch.cat([vlm_cond, phase_cond], dim=-1)  # [B, 2*cond_dim]
        return cond

    def _velocity_step(
        self,
        x_flat: torch.Tensor,    # [B, K*A]
        cond: torch.Tensor,      # [B, 2*cond_dim]
        t: torch.Tensor,         # [B] float in [0, 1]
    ) -> torch.Tensor:
        """单步 velocity 预测：输入 x_t flat + cond + continuous time → 预测速度 flat"""
        t_emb_fp32 = self._sinusoidal_time_embed(t)                     # [B, time_dim] fp32
        t_emb = self.time_mlp(t_emb_fp32.to(cond.dtype))                # [B, time_dim] cond.dtype

        net_in = torch.cat([x_flat, cond, t_emb], dim=-1)
        return self.velocity_net(net_in)                                 # [B, K*A]

    def forward(
        self,
        vlm_hidden: torch.Tensor,
        phase_token: torch.Tensor,
        memory_pad_mask: Optional[torch.Tensor] = None,
        action_gt: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        训练模式（action_gt 提供）:
            返回 {"chunk_loss": velocity_mse, "action_chunk": action_gt 占位}
        推理模式（action_gt = None）:
            返回 {"chunk_loss": None, "action_chunk": Euler 积分结果}
        """
        B = vlm_hidden.shape[0]
        device = vlm_hidden.device

        cond = self._get_cond(vlm_hidden, phase_token, memory_pad_mask)

        if action_gt is not None:
            # ════════════════════════════════════════════════════════════
            # 训练 / eval 监控：conditional flow matching loss
            # ════════════════════════════════════════════════════════════
            action_gt_fp32 = action_gt.float()                              # [B, K, A]

            # 随机采样连续时间 t ∈ [0, 1]
            t = torch.rand(B, device=device, dtype=torch.float32)            # [B]

            # 采样高斯噪声作为路径起点
            z = torch.randn_like(action_gt_fp32)                             # [B, K, A] fp32

            # 构造直线路径：x_t = (1-t) * z + t * action_gt
            # t shape [B] → [B, 1, 1] 广播
            t_expand = t.view(B, 1, 1)
            x_t = (1.0 - t_expand) * z + t_expand * action_gt_fp32           # [B, K, A] fp32

            # 目标速度：dx/dt = action_gt - z（直线路径上的常速度）
            v_target = action_gt_fp32 - z                                    # [B, K, A] fp32

            # 转回 cond 的 dtype 给 velocity network
            x_t_flat = x_t.flatten(1).to(cond.dtype)                         # [B, K*A]

            # 预测速度
            v_pred_flat = self._velocity_step(x_t_flat, cond, t)
            v_pred = v_pred_flat.view(B, self.chunk_size, self.action_dim)

            # MSE loss in fp32
            loss = F.mse_loss(v_pred.float(), v_target)

            return {
                "chunk_loss": loss,
                "action_chunk": action_gt,  # passthrough，便于训练日志
            }
        else:
            # ════════════════════════════════════════════════════════════
            # 推理：Euler 积分 t: 0 → 1
            # ════════════════════════════════════════════════════════════
            N = self.num_inference_steps
            dt = 1.0 / N

            # 从随机噪声开始（t=0 端点）
            x = torch.randn(
                B, self.chunk_size, self.action_dim,
                device=device, dtype=torch.float32,
            )

            # Euler 积分：x_{t+dt} = x_t + dt * v(x_t, t)
            # 使用离散的 time steps: t_i = i * dt, i = 0, 1, ..., N-1
            for i in range(N):
                t_scalar = i * dt
                t_batch = torch.full(
                    (B,), t_scalar, device=device, dtype=torch.float32
                )

                # 转 cond 的 dtype 给 velocity network
                x_flat = x.flatten(1).to(cond.dtype)
                v_pred_flat = self._velocity_step(x_flat, cond, t_batch)
                v_pred = v_pred_flat.view(
                    B, self.chunk_size, self.action_dim
                ).float()

                # Euler step in fp32
                x = x + dt * v_pred

            return {
                "chunk_loss": None,
                "action_chunk": x.to(vlm_hidden.dtype),
            }


# ─────────────────────────────────────────────────────────────────────────────
# MagVLAPolicy
# ─────────────────────────────────────────────────────────────────────────────
class MagVLAPolicy(nn.Module):
    """
    完整 MagVLA Policy，包含:
        - VLM Backbone (Qwen2.5-VL-7B + LoRA)
        - State Conditioning: 机器人状态 Linear(4, D) → state token，拼入 VLM hidden
        - Phase Head (2 分类: approach vs transport，with masked mean pooling + state + motion 输入)
        - Flow Matching Chunk Head (conditional flow matching 小脑)

    关键修改 (v13-flow):
        - DiffusionChunkHead → FlowMatchingChunkHead（Euler 积分）
        - 不再依赖 diffusers 库
        - chunk_loss 改为 velocity prediction MSE（在 head 内部计算）
        - 训练时 head 构造直线路径并预测 velocity
        - 推理时 head 跑 Euler 积分（10 步）生成动作
    """

    def __init__(
        self,
        vlm_name: str = config.VLM_NAME,
        chunk_size: int = config.CHUNK_SIZE,
        image_history: int = config.IMAGE_HISTORY,
        image_downscale: Optional[Tuple[int, int]] = config.IMAGE_DOWNSCALE,
        use_lora: bool = config.USE_LORA,
        device: str = config.DEVICE,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.image_history = image_history
        self.image_downscale = image_downscale

        # ── 1. 加载 VLM backbone ───────────────────────────────────────
        model_config = Qwen2_5_VLConfig.from_pretrained(vlm_name)
        model_config.vision_config.hidden_size = 1280

        self.processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True)

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            config=model_config,
            torch_dtype=config.VLM_DTYPE,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        text_cfg = self.vlm.config.get_text_config()
        hidden_dim = text_cfg.hidden_size if hasattr(text_cfg, 'hidden_size') else text_cfg.hidden_dim

        # ── 2. LoRA ───────────────────────────────────────────────────
        if use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=config.LORA_TARGET_MODULES,
            )
            self.vlm = get_peft_model(self.vlm, lora_cfg)
            self.vlm.print_trainable_parameters()

        # Freeze 非 LoRA 参数
        for name, param in self.vlm.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        # ── 3. 辅助 heads ─────────────────────────────────────────────
        self.state_proj = nn.Linear(4, hidden_dim)
        self.phase_head = PhaseHead(hidden_dim=hidden_dim, state_dim=4, num_phases=2)
        # 注意：变量名仍叫 act_head 以保持和 train_ddp.py / inference 代码兼容，
        # 但实际是 FlowMatchingChunkHead 实例（conditional flow matching 小脑）
        self.act_head = FlowMatchingChunkHead(
            hidden_dim=hidden_dim,
            action_dim=4,
            chunk_size=chunk_size,
            num_inference_steps=config.FLOW_INFERENCE_STEPS,
            time_scale=config.FLOW_TIME_SCALE,
        )

        # ── 4. 移动到 GPU ───────────────────────────────────────────────
        self.state_proj = self.state_proj.to(device, dtype=config.VLM_DTYPE)
        self.phase_head = self.phase_head.to(device, dtype=config.VLM_DTYPE)
        self.act_head = self.act_head.to(device, dtype=config.VLM_DTYPE)

        self._device = device

    def _get_device(self) -> str:
        """动态获取模型所在的设备"""
        return next(self.vlm.parameters()).device

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        if self.image_downscale is not None:
            import cv2
            h, w = self.image_downscale
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def _run_vlm(self, images: torch.Tensor, task_texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VLM 前向传播：返回 (vlm_hidden, vlm_pad_mask)

        images: [B, T, H, W, 3] uint8 tensor，RGB
        task_texts: list of str, len = B

        返回:
            vlm_hidden: [B, L, D]
            vlm_pad_mask: [B, L] BoolTensor, True=padding, False=valid
        """
        B, T = images.shape[0], images.shape[1]

        hidden_states_list = []
        original_lengths = []
        for b in range(B):
            # ── 将 4 帧转换为 PIL Images ─────────────────────────────────
            pil_images = []
            for t in range(T):
                img_np = images[b, t].cpu().numpy()
                img_np = self._resize_image(img_np)
                pil_img = Image.fromarray(img_np.astype(np.uint8))
                pil_images.append(pil_img)

            # ── 构造多图 message ─────────────────────────────────────────
            content = [{"type": "image"} for _ in range(T)]
            content.append({"type": "text", "text": task_texts[b]})
            message = [{"role": "user", "content": content}]

            text_input = self.processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False,
            )
            inputs = self.processor(
                text=[text_input], images=pil_images, padding=False, return_tensors="pt",
            )

            vlm_inputs = {
                "input_ids": inputs["input_ids"].to(self._get_device()),
                "pixel_values": inputs["pixel_values"].to(self._get_device()),
                "image_grid_thw": inputs["image_grid_thw"].to(self._get_device()),
            }
            if "mm_token_type_ids" in inputs:
                vlm_inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(self._get_device())

            with torch.amp.autocast("cuda", dtype=config.VLM_DTYPE):
                outputs = self.vlm(**vlm_inputs, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states[-1]  # [1, L_b, D]
            hidden_states_list.append(hs)
            original_lengths.append(hs.shape[1])

        # ── Padding 到统一长度，并构造 mask ─────────────────────────────
        if B == 1:
            return hidden_states_list[0], torch.zeros(
                (1, hidden_states_list[0].shape[1]),
                dtype=torch.bool,
                device=self._get_device(),
            )
        else:
            max_len = max(hs.shape[1] for hs in hidden_states_list)
            D = hidden_states_list[0].shape[2]
            padded_list = []
            mask_list = []
            for b, (hs, orig_len) in enumerate(zip(hidden_states_list, original_lengths)):
                pad_len = max_len - orig_len
                if pad_len > 0:
                    pad = torch.zeros(1, pad_len, D, device=hs.device, dtype=hs.dtype)
                    hs = torch.cat([hs, pad], dim=1)
                    # True=padding at the END of the sequence
                    mask = torch.zeros(1, max_len, dtype=torch.bool, device=hs.device)
                    mask[:, orig_len:] = True
                else:
                    # No padding: all valid
                    mask = torch.zeros(1, max_len, dtype=torch.bool, device=hs.device)
                padded_list.append(hs)
                mask_list.append(mask)

            vlm_hidden = torch.cat(padded_list, dim=0)          # [B, max_len, D]
            vlm_pad_mask = torch.cat(mask_list, dim=0)          # [B, max_len] True=padding
            return vlm_hidden, vlm_pad_mask

    def _inject_state(
        self,
        vlm_hidden: torch.Tensor,
        vlm_pad_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将机器人状态注入 VLM hidden state 序列，同时更新 mask。

        vlm_hidden:   [B, L, D]
        vlm_pad_mask: [B, L] BoolTensor, True=padding
        state:        [B, 4] 当前双臂 UMP 位置（normalized）

        返回:
            vlm_hidden:   [B, 1+L, D]  state token 拼在最前面
            vlm_pad_mask: [B, 1+L]    state token 位置是 False（valid）
        """
        state_token = self.state_proj(state.to(dtype=self.state_proj.weight.dtype))  # [B, D]
        state_token = state_token.unsqueeze(1)  # [B, 1, D]
        vlm_hidden = torch.cat([state_token, vlm_hidden], dim=1)  # [B, 1+L, D]

        # state token 是真实 token，不是 padding，所以拼 False
        state_valid = torch.zeros(vlm_hidden.shape[0], 1, dtype=torch.bool, device=vlm_hidden.device)
        vlm_pad_mask = torch.cat([state_valid, vlm_pad_mask], dim=1)  # [B, 1+L]

        return vlm_hidden, vlm_pad_mask

    def forward(
        self,
        images: torch.Tensor,
        task_texts: list,
        states: torch.Tensor,
        phase_gt: torch.Tensor,
        return_loss: bool = True,
        action_delta_gt: Optional[torch.Tensor] = None,
        state_history: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        前向传播（走完整 mask 链路，不绕 DDP）。

        images:     [B, T=4, H, W, 3] uint8，RGB
        task_texts: [B] str
        states:     [B, 4] float32，normalized，双臂 UMP 位置
        phase_gt:   [B] long，训练时用于 phase_loss，推理时为 None
        state_history: [B, 4, 4] normalized 4 帧历史状态
        """
        # ── 1. VLM Backbone → (hidden, pad_mask) ───────────────────────
        vlm_hidden, vlm_pad_mask = self._run_vlm(images, task_texts)

        # ── 2. State Conditioning → (hidden, pad_mask) ─────────────────
        vlm_hidden, vlm_pad_mask = self._inject_state(vlm_hidden, vlm_pad_mask, states)

        # ── 3. Phase Head（masked mean pooling）──────────────────────────
        phase_out = self.phase_head(
            vlm_hidden.detach(),
            phase_gt,
            token_pad_mask=vlm_pad_mask,
            state_normalized=states,
            state_history=state_history,
        )
        phase_logits = phase_out["phase_logits"]
        phase_token = phase_out["phase_token"]

        # ── 4. Flow Matching Chunk Head（传入 action_gt 用于训练构造直线路径，None 用于推理 Euler 积分）─
        chunk_out = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            action_gt=action_delta_gt,
        )
        action_chunk = chunk_out["action_chunk"]
        fm_chunk_loss = chunk_out["chunk_loss"]  # velocity MSE，or None 当推理时

        # ── 5. 损失（训练时）──────────────────────────────────────────────
        if return_loss:
            if phase_gt is not None:
                phase_loss = F.cross_entropy(phase_logits, phase_gt)
            else:
                phase_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

            if fm_chunk_loss is not None:
                chunk_loss = fm_chunk_loss
            else:
                chunk_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

            loss = chunk_loss + config.LAMBDA_PHASE * phase_loss

            return {
                "loss": loss,
                "phase_loss": phase_loss,
                "chunk_loss": chunk_loss,
                "phase_logits": phase_logits,
                "action_chunk": action_chunk,
                "phase_id": phase_out["phase_id"],
            }
        else:
            return {
                "phase_logits": phase_logits,
                "action_chunk": action_chunk,
                "phase_id": phase_out["phase_id"],
            }

    def forward_full(
        self,
        images: torch.Tensor,
        task_texts: list,
        states: torch.Tensor,
        phase_gt: torch.Tensor,
        action_delta_gt: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
    ) -> dict:
        """带完整损失计算的前向传播，数据流与 forward() 一致。"""
        # ── 1. VLM Backbone ──────────────────────────────────────────
        vlm_hidden, vlm_pad_mask = self._run_vlm(images, task_texts)

        # ── 2. State Conditioning ──────────────────────────────────────
        vlm_hidden, vlm_pad_mask = self._inject_state(vlm_hidden, vlm_pad_mask, states)

        # ── 3. Phase Head ─────────────────────────────────────────────
        phase_out = self.phase_head(
            vlm_hidden.detach(),
            phase_gt,
            token_pad_mask=vlm_pad_mask,
            state_normalized=states,
            state_history=state_history,
        )
        phase_logits = phase_out["phase_logits"]
        phase_token = phase_out["phase_token"]

        # ── 4. Diffusion Chunk Head ──────────────────────────────────────
        chunk_out = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            action_gt=action_delta_gt,
        )
        action_chunk = chunk_out["action_chunk"]
        fm_chunk_loss = chunk_out["chunk_loss"]  # velocity prediction MSE

        # ── 5. 完整损失 ────────────────────────────────────────────────
        phase_loss = F.cross_entropy(phase_logits, phase_gt)
        chunk_loss = fm_chunk_loss
        loss = chunk_loss + config.LAMBDA_PHASE * phase_loss

        return {
            "loss": loss,
            "phase_loss": phase_loss,
            "chunk_loss": chunk_loss,
            "phase_logits": phase_logits,
            "action_chunk": action_chunk,
            "phase_id": phase_out["phase_id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────
def build_policy(device: str = config.DEVICE) -> MagVLAPolicy:
    policy = MagVLAPolicy(
        vlm_name=config.VLM_NAME,
        chunk_size=config.CHUNK_SIZE,
        image_history=config.IMAGE_HISTORY,
        image_downscale=config.IMAGE_DOWNSCALE,
        use_lora=config.USE_LORA,
        device=device,
    )
    return policy


def get_optimizer(policy: MagVLAPolicy, total_steps: int):
    """分别优化 VLM (LoRA) 和两个 Head"""
    lora_params = [p for n, p in policy.vlm.named_parameters() if "lora" in n.lower()]

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": config.LR, "weight_decay": config.WEIGHT_DECAY},
            {
                "params": policy.phase_head.parameters(),
                "lr": config.LR_PHASE_HEAD,
                "weight_decay": config.WEIGHT_DECAY,
            },
            {
                "params": policy.act_head.parameters(),
                "lr": config.LR_ACT_HEAD,
                "weight_decay": config.WEIGHT_DECAY,
            },
            {
                "params": policy.state_proj.parameters(),
                "lr": config.LR_ACT_HEAD,
                "weight_decay": config.WEIGHT_DECAY,
            },
        ]
    )

    # Warmup + Plateau + Cosine decay（不归零）
    import math

    def lr_lambda(step):
        warmup = config.WARMUP_STEPS
        plateau = config.PLATEAU_STEPS
        min_ratio = config.MIN_LR_RATIO

        if step < warmup:
            return float(step) / max(1, warmup)

        if step < warmup + plateau:
            return 1.0

        remain = max(1, total_steps - warmup - plateau)
        progress = min(max((step - warmup - plateau) / remain, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
