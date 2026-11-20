"""
MagVLA Policy (Version 13 - 2-Phase + Direct Prompt Injection to ACT Cerebellum)

关键修改（相对 v13 原版）：
- PhaseHead num_phases 从 3 改为 2（approach=0 vs transport=1）
- 删除 return 阶段（phase=2）的数据帧
- **新增直接 prompt 注入路径**：将 task_text 的 token 同时输入到 ACT 小脑模型
  - 启动时用 frozen DistilBERT 编码所有 task_text → 缓存到 self._text_cache
  - 用 trainable Linear(768, hidden_dim) 投影到 LLM hidden 空间 → text_token
  - text_token 作为 ACT cross-attn memory 的额外 token，拼在最前面
- 动机：v13 真机实验中，prompt 信息被 ~1024 vision tokens 稀释（vs ~30 text tokens），
  导致小脑无法区分 task A/B/C。绕开 VLM 中的视觉 bottleneck，直接给小脑提供任务描述
- DistilBERT 仅在 build_text_cache() 调用一次，之后被释放，不占用显存，DDP 也看不到它
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
# ACT Chunk Head（手动 MHA，完全控制 padding mask + text token 注入）
# ─────────────────────────────────────────────────────────────────────────────
class ACTChunkHead(nn.Module):
    """
    输入:
        vlm_hidden: [B, L, D]       # VLM backbone 输出
        phase_token: [B, D]         # Phase Head 输出的条件向量
        memory_pad_mask: [B, L]     # BoolTensor, True=padding, False=valid
                                   # （已含 phase_token 位置的 False）
        text_token: [B, D] or None  # 直接注入的 prompt 文本特征（可选）
                                   #   - 如果不为 None，会作为 memory 的第 0 个 token 加入
                                   #   - cross-attention 可以同时关注 text、phase、vlm hidden
                                   #   - 用途：让小脑直接接收"任务提示"信号，
                                   #     绕开 VLM 中视觉 token 主导导致的 prompt 信息被冲淡问题
                                   #     (~1024 vision tokens vs ~30 text tokens)
    输出:
        action_chunk: [B, K, 4]     # 未来 K 步的 action_delta 预测

    实现说明：
    - 不使用 nn.MultiheadAttention（key_padding_mask 有维度校验问题）
    - 使用手动 QKV 投影 + F.scaled_dot_product_attention
    - self-attn 不需要 padding mask（queries 长度固定）
    - cross-attn 只对 memory 施 padding mask
    - text_token 与 phase_token 同样作为单 token 拼到 memory 最前面
    """

    def __init__(self, hidden_dim: int = 3584, chunk_size: int = 5, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.nhead = 8
        self.head_dim = hidden_dim // self.nhead  # 448

        # 可学习的 action query（DERT 风格）
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)

        # 每层：LayerNorm → Self-Attn → LayerNorm → Cross-Attn → FFN
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Self-attn: Q=K=V=queries
                "ln1": nn.LayerNorm(hidden_dim),
                "wq1": nn.Linear(hidden_dim, hidden_dim),
                "wk1": nn.Linear(hidden_dim, hidden_dim),
                "wv1": nn.Linear(hidden_dim, hidden_dim),
                "out1": nn.Linear(hidden_dim, hidden_dim),
                # Cross-attn: Q=queries, K=V=memory
                "ln2": nn.LayerNorm(hidden_dim),
                "wq2": nn.Linear(hidden_dim, hidden_dim),
                "wk2": nn.Linear(hidden_dim, hidden_dim),
                "wv2": nn.Linear(hidden_dim, hidden_dim),
                "out2": nn.Linear(hidden_dim, hidden_dim),
                # FFN
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(0.1),
                ),
            })
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_dim)

        # 投影到动作空间
        self.action_proj = nn.Linear(hidden_dim, 4)
        nn.init.xavier_uniform_(self.action_proj.weight, gain=0.01)
        nn.init.zeros_(self.action_proj.bias)

    def _attn(
        self,
        q: torch.Tensor,      # [B, T, D]
        k: torch.Tensor,      # [B, S, D]
        v: torch.Tensor,      # [B, S, D]
        pad_mask: Optional[torch.Tensor] = None,  # [B, S] BoolTensor, True=masked
    ) -> torch.Tensor:
        """
        手动 multi-head attention + scaled dot-product attention。
        pad_mask 只应用于 K/V 方向，Q 始终不被 mask。
        """
        B, T, D = q.shape
        _, S, _ = k.shape
        h = self.nhead
        dk = D // h

        # Reshape: [B, T, D] → [B, h, T, dk]
        q = q.view(B, T, h, dk).transpose(1, 2)     # [B, h, T, dk]
        k = k.view(B, S, h, dk).transpose(1, 2)     # [B, h, S, dk]
        v = v.view(B, S, h, dk).transpose(1, 2)     # [B, h, S, dk]

        # Scaled dot-product
        scale = dk ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, h, T, S]

        if pad_mask is not None:
            # pad_mask is [B, S] where S == attn.shape[3]
            # True = masked out, broadcast to [B, 1, 1, S]
            attn = attn.masked_fill(pad_mask.view(B, 1, 1, S), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, h, T, dk]

        # Merge heads: [B, h, T, dk] → [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out

    def forward(
        self,
        vlm_hidden: torch.Tensor,
        phase_token: torch.Tensor,
        memory_pad_mask: Optional[torch.Tensor] = None,
        text_token: Optional[torch.Tensor] = None,
    ):
        """
        vlm_hidden:  [B, L, D]
        phase_token: [B, D]
        memory_pad_mask: [B, L] True=padding（已含 state token 位置的 False）
        text_token:  [B, D] 或 None（如不为 None，作为额外的 memory token 拼到最前面）

        Memory 拼接顺序：[text_token?, phase_token, vlm_hidden]
        """
        B, L, D = vlm_hidden.shape

        # ── 构造 memory ─────────────────────────────────────────────────
        phase_token_seq = phase_token.unsqueeze(1)  # [B, 1, D]
        if text_token is not None:
            text_token_seq = text_token.unsqueeze(1)  # [B, 1, D]
            memory = torch.cat([text_token_seq, phase_token_seq, vlm_hidden], dim=1)  # [B, 2+L, D]
            n_extra = 2  # text + phase
        else:
            memory = torch.cat([phase_token_seq, vlm_hidden], dim=1)  # [B, 1+L, D]
            n_extra = 1  # phase only

        # ── memory_pad_mask 拼接（额外的 token 位置都是 valid）─────────────
        if memory_pad_mask is not None:
            extra_false = torch.zeros(B, n_extra, dtype=torch.bool, device=vlm_hidden.device)
            mem_pad = torch.cat([extra_false, memory_pad_mask], dim=1)  # [B, n_extra+L]
        else:
            mem_pad = torch.zeros(B, n_extra + L, dtype=torch.bool, device=vlm_hidden.device)

        # ── 构造 queries ───────────────────────────────────────────────
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # ── 多层 transformer blocks ─────────────────────────────────────
        for layer in self.layers:
            q = queries

            # Self-attention（queries 长度固定 K，无 padding）
            q_ln = layer["ln1"](q)
            attn_out = self._attn(layer["wq1"](q_ln), layer["wk1"](q_ln), layer["wv1"](q_ln))
            queries = q + layer["out1"](attn_out)

            # Cross-attention（queries → memory，使用 padding mask）
            q_ln = layer["ln2"](queries)
            cross_out = self._attn(
                layer["wq2"](q_ln),
                layer["wk2"](memory),
                layer["wv2"](memory),
                pad_mask=mem_pad,
            )
            queries = queries + layer["out2"](cross_out)

            # FFN
            queries = queries + layer["ffn"](queries)

        queries = self.final_ln(queries)

        # ── 投影到动作空间 ───────────────────────────────────────────────
        action_chunk = self.action_proj(queries)  # [B, K, 4]

        return action_chunk


# ─────────────────────────────────────────────────────────────────────────────
# MagVLAPolicy
# ─────────────────────────────────────────────────────────────────────────────
class MagVLAPolicy(nn.Module):
    """
    完整 MagVLA Policy，包含:
        - VLM Backbone (Qwen2.5-VL-7B + LoRA)
        - State Conditioning: 机器人状态 Linear(4, D) → state token，拼入 VLM hidden
        - Phase Head (2 分类: approach vs transport，with masked mean pooling + state + motion 输入)
        - **Text Encoder (frozen DistilBERT, 用于直接注入 prompt 到 ACT 小脑)**
        - ACT Chunk Head (with manual MHA padding mask + 可选 text_token 注入)

    关键修改 (v13-text，相对 v13 原版):
        - 新增直接 prompt 注入路径：
          1. 启动时用 frozen DistilBERT 编码所有 task_text → 缓存到 self._text_cache
          2. 用 trainable Linear(768, hidden_dim) 投影到 LLM hidden 空间 → text_token
          3. text_token 作为 ACT cross-attn memory 的额外 token
        - 动机：真机实验中观察到任务描述信息被 ~1024 个 vision tokens
          稀释（vs 30 个 text tokens），导致小脑无法区分 task A/B/C
        - 设计：DistilBERT 仅在 build_text_cache() 调用一次，之后就被释放
          → 训练/推理时无 DistilBERT 开销，DDP 也看不到它
        - 使用：在 DDP wrap 之前调用 policy.build_text_cache(list(train_ds.tasks.values()))
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
            device_map={"": device},
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
        self.act_head = ACTChunkHead(hidden_dim=hidden_dim, chunk_size=chunk_size, num_layers=2)

        # ── 3.5 Text injection 路径 ────────────────────────────────────
        # text_proj: DistilBERT 768 → hidden_dim
        # _text_cache: dict[str → tensor[hidden_dim] on device], 由 build_text_cache() 填充
        # _distilbert / _distilbert_tok: 临时持有，build_text_cache() 后被释放
        self.text_feat_dim = 768  # DistilBERT base hidden size
        self.text_proj = nn.Linear(self.text_feat_dim, hidden_dim)
        # 用 plain dict 而非 ParameterDict，避免被 DDP 当做参数
        self._text_cache: dict = {}

        # ── 4. 移动到 GPU ───────────────────────────────────────────────
        self.state_proj = self.state_proj.to(device, dtype=config.VLM_DTYPE)
        self.phase_head = self.phase_head.to(device, dtype=config.VLM_DTYPE)
        self.act_head = self.act_head.to(device, dtype=config.VLM_DTYPE)
        self.text_proj = self.text_proj.to(device, dtype=config.VLM_DTYPE)

        self._device = device
        self.hidden_dim = hidden_dim

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

    # ─────────────────────────────────────────────────────────────────
    # Text injection 路径（直接将 prompt 注入 ACT 小脑）
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def build_text_cache(self, texts: list, batch_size: int = 16, verbose: bool = True):
        """
        预先用 frozen DistilBERT 编码所有可能出现的 task_texts，缓存到 self._text_cache。

        必须在以下两个时机调用之一（任选其一即可）：
        1. 训练前：在 DDP wrap 之前从 train_ds.tasks 收集所有 unique prompts 后调用
        2. 推理前：在 load_state_dict 之后从测试集 / config 已知 prompts 调用

        DistilBERT 在调用结束后会被释放（del + cuda.empty_cache），不占用任何显存，
        DDP 也看不到它，所以 find_unused_parameters 可以保持 False。

        参数：
            texts: 所有可能出现的 task_text 列表（重复无害，会自动 dedup）
            batch_size: DistilBERT 编码 batch 大小（3 个 prompts 一次就够了）
        """
        # 取 unique，跳过已缓存的
        unique = sorted({t for t in texts if t not in self._text_cache})
        if len(unique) == 0:
            if verbose:
                print(f"  [text_cache] all {len(texts)} texts already cached, skipping")
            return

        if verbose:
            print(f"  [text_cache] building DistilBERT cache for {len(unique)} unique prompt(s)...")
            for i, t in enumerate(unique):
                preview = t if len(t) < 80 else t[:77] + "..."
                print(f"    [{i}] '{preview}'")

        from transformers import DistilBertModel, DistilBertTokenizerFast

        # 加载 frozen DistilBERT 到当前设备
        tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        bert = bert.to(self._get_device()).eval()
        # 冻结所有参数（虽然反正会被 del，但加 requires_grad=False 防止意外触发 backward）
        for p in bert.parameters():
            p.requires_grad = False

        # 分批编码
        for i in range(0, len(unique), batch_size):
            chunk = unique[i : i + batch_size]
            inputs = tok(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            inputs = {k: v.to(self._get_device()) for k, v in inputs.items()}
            out = bert(**inputs)
            # mean pool over valid tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            # [b, 768] fp32（DistilBERT 默认 fp32）
            for j, t in enumerate(chunk):
                # 转为 VLM_DTYPE 并保留在 device 上，避免每次 forward 移动
                self._text_cache[t] = pooled[j].detach().to(dtype=config.VLM_DTYPE)

        # 显式释放 DistilBERT
        del bert, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print(f"  [text_cache] cached {len(self._text_cache)} prompt(s) total, DistilBERT released")

    def _get_text_token(self, task_texts: list) -> torch.Tensor:
        """
        从缓存中查找每个 task_text 的 768-d 特征，stack 后投影到 hidden_dim。

        参数：
            task_texts: list[str] of length B
        返回：
            text_token: [B, hidden_dim]，dtype 与 text_proj 一致
        """
        missing = [t for t in task_texts if t not in self._text_cache]
        if missing:
            raise RuntimeError(
                f"text cache miss: {len(missing)} prompts not in cache. "
                f"First missing: '{missing[0][:80]}'. "
                f"Call policy.build_text_cache(list(train_ds.tasks.values())) "
                f"after building dataset and before DDP wrap."
            )
        feats = torch.stack(
            [self._text_cache[t] for t in task_texts], dim=0
        )  # [B, 768] on device, VLM_DTYPE
        text_token = self.text_proj(feats)  # [B, hidden_dim]
        return text_token

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

        # ── 3.5 Text token（直接 prompt 注入路径）─────────────────────────
        # 从缓存查找每个 task_text 的 DistilBERT 特征 → 投影到 hidden_dim
        text_token = self._get_text_token(task_texts)  # [B, hidden_dim]

        # ── 4. ACT Chunk Head（传入 padding mask + text_token）────────────
        action_chunk = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            text_token=text_token,
        )

        # ── 5. 损失（训练时）──────────────────────────────────────────────
        if return_loss:
            if phase_gt is not None:
                phase_loss = F.cross_entropy(phase_logits, phase_gt)
            else:
                phase_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

            if action_delta_gt is not None:
                chunk_loss = F.smooth_l1_loss(action_chunk, action_delta_gt, beta=1.0)
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

        # ── 3.5 Text token（直接 prompt 注入路径）─────────────────────────
        text_token = self._get_text_token(task_texts)  # [B, hidden_dim]

        # ── 4. ACT Chunk Head ──────────────────────────────────────────
        action_chunk = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            text_token=text_token,
        )

        # ── 5. 完整损失 ───────────────────────────────────────────────
        phase_loss = F.cross_entropy(phase_logits, phase_gt)
        chunk_loss = F.smooth_l1_loss(action_chunk, action_delta_gt, beta=1.0)
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
    """分别优化 VLM (LoRA)、两个 Head 和 text_proj"""
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
            {
                "params": policy.text_proj.parameters(),
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
