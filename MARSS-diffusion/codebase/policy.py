"""
MagVLA Policy (Version 13 - 2-Phase: 删除 return，PhaseHead 改为 2 分类)

关键修改（相对 v11）：
- PhaseHead num_phases 从 3 改为 2（approach=0 vs transport=1）
- 删除 return 阶段（phase=2）的数据帧
- 解决 approach↔return 混淆问题（因为 return 根本不存在了）
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

# diffusers 用于 Diffusion Policy 的 noise scheduler
# pip install diffusers
try:
    from diffusers import DDIMScheduler
except ImportError as e:
    raise ImportError(
        "diffusers 未安装。请运行: pip install diffusers"
    ) from e

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
        if phase_gt is not None:
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
# Diffusion Chunk Head（DDIM-based diffusion policy 小脑）
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionChunkHead(nn.Module):
    """
    输入:
        vlm_hidden: [B, L, D]       # VLM backbone 输出
        phase_token: [B, D]         # Phase Head 输出的条件向量
        memory_pad_mask: [B, L]     # BoolTensor, True=padding
        action_gt: [B, K, A] or None
                                   # 训练时传入 → 计算 noise 预测 MSE 损失
                                   # 推理时传 None → 跑 DDIM 采样

    输出 (dict):
        chunk_loss: scalar tensor or None
        action_chunk: [B, K, A]    # 训练时 = action_gt（passthrough）
                                   # 推理时 = DDIM 采样的动作

    设计说明：
    - 用 condition vector（pooled VLM hidden + phase_token）作为 denoiser 的 global condition
    - Denoiser 是简单的 MLP（动作维度 = K*A = 20，不需要 U-Net）
    - 训练：epsilon prediction（预测加入的高斯噪声），MSE loss
    - 推理：DDIM 采样，用 num_inference_steps 步快速去噪
    - dtype 处理：noise schedule 用 fp32（数值稳定），denoiser MLP 跟随 vlm_hidden 的 dtype（bf16）
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        action_dim: int = 4,
        chunk_size: int = 5,
        cond_dim: int = 256,
        time_dim: int = 128,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        action_flat = chunk_size * action_dim  # 5 * 4 = 20

        # ── 条件投影：vlm_hidden 池化 + phase_token ─────────────────────
        self.vlm_proj = nn.Linear(hidden_dim, cond_dim)
        self.phase_proj = nn.Linear(hidden_dim, cond_dim)

        # ── 时间 embedding（sinusoidal → MLP）────────────────────────────
        # sinusoidal 部分在 forward 中计算（不需要可学习参数）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── Denoiser MLP（输入：noisy action + cond + time_emb）─────────
        denoiser_in = action_flat + cond_dim * 2 + time_dim
        denoiser_hidden = 512
        self.denoiser = nn.Sequential(
            nn.Linear(denoiser_in, denoiser_hidden),
            nn.SiLU(),
            nn.Linear(denoiser_hidden, denoiser_hidden),
            nn.SiLU(),
            nn.Linear(denoiser_hidden, denoiser_hidden),
            nn.SiLU(),
            nn.Linear(denoiser_hidden, action_flat),
        )
        # 输出层初始化为 0，让初始预测接近 0（标准 diffusion 实践）
        nn.init.zeros_(self.denoiser[-1].weight)
        nn.init.zeros_(self.denoiser[-1].bias)

        # ── DDIM noise scheduler ───────────────────────────────────────
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",   # cosine schedule（DDPM++ 标准）
            clip_sample=False,                    # action_delta 已 clip 到 [-5, 5]，不再 clip
            prediction_type="epsilon",            # 预测噪声（最稳健的目标）
        )

        # 兼容性占位：和 ACT head 接口一致（虽然 diffusion 不用 action_queries）
        # 也提供 phase_embedding 接口，便于推理代码复用
        self.phase_embedding = None  # diffusion 不需要单独的 phase_embedding

    def _sinusoidal_time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding for diffusion timesteps.
        t: [B] long
        returns: [B, time_dim] fp32
        """
        device = t.device
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, dtype=torch.float32, device=device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
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
        phase_cond = self.phase_proj(phase_token)    # [B, cond_dim]
        cond = torch.cat([vlm_cond, phase_cond], dim=-1)  # [B, 2*cond_dim]
        return cond

    def _denoise_step(
        self,
        x_flat: torch.Tensor,    # [B, K*A]
        cond: torch.Tensor,       # [B, 2*cond_dim]
        timesteps: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """单步 denoise：输入 noisy action flat + cond + timestep → 预测 noise flat"""
        # Sinusoidal time embed → MLP（time_mlp 在 cond 的 dtype）
        t_emb_fp32 = self._sinusoidal_time_embed(timesteps)             # [B, time_dim] fp32
        t_emb = self.time_mlp(t_emb_fp32.to(cond.dtype))                # [B, time_dim] cond.dtype

        denoiser_in = torch.cat([x_flat, cond, t_emb], dim=-1)
        return self.denoiser(denoiser_in)                                # [B, K*A]

    def forward(
        self,
        vlm_hidden: torch.Tensor,
        phase_token: torch.Tensor,
        memory_pad_mask: Optional[torch.Tensor] = None,
        action_gt: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        训练模式（action_gt 提供）:
            返回 {"chunk_loss": noise_mse, "action_chunk": action_gt 占位}
        推理模式（action_gt = None）:
            返回 {"chunk_loss": None, "action_chunk": DDIM 采样结果}
        """
        B = vlm_hidden.shape[0]
        device = vlm_hidden.device

        cond = self._get_cond(vlm_hidden, phase_token, memory_pad_mask)

        if action_gt is not None:
            # ════════════════════════════════════════════════════════════
            # 训练 / eval 监控：noise prediction loss
            # ════════════════════════════════════════════════════════════
            # 转 fp32 保证 noise schedule 数值稳定
            action_gt_fp32 = action_gt.float()                              # [B, K, A]

            # 随机采样 timesteps
            timesteps = torch.randint(
                0, self.num_train_timesteps, (B,),
                device=device, dtype=torch.long,
            )

            # 采样高斯噪声
            noise = torch.randn_like(action_gt_fp32)                        # [B, K, A] fp32

            # 加噪：q(x_t | x_0)
            noisy_action = self.scheduler.add_noise(
                action_gt_fp32, noise, timesteps
            )                                                                # [B, K, A] fp32

            # 转回 cond 的 dtype 给 denoiser
            noisy_flat = noisy_action.flatten(1).to(cond.dtype)             # [B, K*A]

            # 预测噪声
            noise_pred_flat = self._denoise_step(noisy_flat, cond, timesteps)
            noise_pred = noise_pred_flat.view(B, self.chunk_size, self.action_dim)

            # MSE loss in fp32
            loss = F.mse_loss(noise_pred.float(), noise)

            return {
                "chunk_loss": loss,
                "action_chunk": action_gt,  # passthrough，便于训练日志
            }
        else:
            # ════════════════════════════════════════════════════════════
            # 推理：DDIM 采样
            # ════════════════════════════════════════════════════════════
            self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            # 从随机噪声开始
            x = torch.randn(
                B, self.chunk_size, self.action_dim,
                device=device, dtype=torch.float32,
            )

            for t in self.scheduler.timesteps:
                t_batch = t.expand(B).to(device=device, dtype=torch.long)

                # 转 cond 的 dtype 给 denoiser
                x_flat = x.flatten(1).to(cond.dtype)
                noise_pred_flat = self._denoise_step(x_flat, cond, t_batch)
                noise_pred = noise_pred_flat.view(
                    B, self.chunk_size, self.action_dim
                ).float()

                # Scheduler step in fp32
                x = self.scheduler.step(noise_pred, t, x).prev_sample

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
        - Diffusion Chunk Head (DDIM-based diffusion policy 小脑)

    关键修改 (v14):
        - ACT Chunk Head → DiffusionChunkHead（DDIM 采样）
        - chunk_loss 改为 noise prediction MSE（在 head 内部计算）
        - 训练时 head 接收 action_gt 加噪并预测噪声
        - 推理时 head 跑 DDIM 采样（10 步）生成动作
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
        # 但实际是 DiffusionChunkHead 实例（DDIM-based diffusion policy 小脑）
        self.act_head = DiffusionChunkHead(
            hidden_dim=hidden_dim,
            action_dim=4,
            chunk_size=chunk_size,
            num_train_timesteps=config.DIFFUSION_TRAIN_TIMESTEPS,
            num_inference_steps=config.DIFFUSION_INFERENCE_STEPS,
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

        # ── 4. Diffusion Chunk Head（传入 action_gt 用于训练加噪，None 用于推理 DDIM 采样）─
        chunk_out = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            action_gt=action_delta_gt,
        )
        action_chunk = chunk_out["action_chunk"]
        diffusion_chunk_loss = chunk_out["chunk_loss"]  # noise MSE，or None 当推理时

        # ── 5. 损失（训练时）──────────────────────────────────────────────
        if return_loss:
            if phase_gt is not None:
                phase_loss = F.cross_entropy(phase_logits, phase_gt)
            else:
                phase_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

            if diffusion_chunk_loss is not None:
                chunk_loss = diffusion_chunk_loss
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
        diffusion_chunk_loss = chunk_out["chunk_loss"]  # noise prediction MSE

        # ── 5. 完整损失 ────────────────────────────────────────────────
        phase_loss = F.cross_entropy(phase_logits, phase_gt)
        chunk_loss = diffusion_chunk_loss
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
