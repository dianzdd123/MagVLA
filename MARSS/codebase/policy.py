"""Mag-VLA implementation for phase-conditioned ACT. Action chunks describe planar
increments for the left and right magnet-mounted arms.
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
    """Estimate approach or transport from pooled multimodal memory, the current arm
    state and displacement over the history window. The selected phase indexes a
    learned embedding for action decoding.
    """

    def __init__(self, hidden_dim: int = 3584, state_dim: int = 4, num_phases: int = 2):
        super().__init__()
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim

        # Project the current planar arm positions.
        self.state_proj_phase = nn.Linear(state_dim, 64)

        # Project the displacement across the state-history window.
        self.motion_proj = nn.Linear(state_dim, 64)

        # Concatenate pooled memory with 64 position and 64 motion features.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_phases),
        )

        # Embed the selected phase as an action-conditioning token.
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
        """Pool valid tokens in [B, L, D] memory. True mask entries denote padding.
        Return two-class logits, the selected phase ID and its [B, D] embedding. A
        supplied label selects the embedding only in training mode.
        """
        B, L, D = vlm_hidden.shape
        dtype = vlm_hidden.dtype

        # Average valid memory tokens, excluding padding.
        if token_pad_mask is not None:
            valid_mask = (~token_pad_mask).unsqueeze(-1).to(dtype)
            valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
            vision_pool = (vlm_hidden * valid_mask).sum(dim=1) / valid_count
        else:
            vision_pool = vlm_hidden.mean(dim=1)

        # Project the current planar arm positions before phase classification.
        if state_normalized is not None:
            state_feat = self.state_proj_phase(state_normalized.to(dtype))
        else:
            state_feat = torch.zeros(B, 64, device=vlm_hidden.device, dtype=dtype)

        # Subtract the oldest state from the newest to estimate displacement.
        if state_history is not None:
            motion = state_history[:, -1, :] - state_history[:, 0, :]
            motion_feat = self.motion_proj(motion.to(dtype))
        else:
            motion_feat = torch.zeros(B, 64, device=vlm_hidden.device, dtype=dtype)

        # Concatenate pooled memory, position and motion before phase classification.
        combined = torch.cat([vision_pool, state_feat, motion_feat], dim=-1)
        phase_logits = self.classifier(combined)

        # Use labelled phase in training when available and predicted phase in evaluation.
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
# ACT attention uses explicit key-padding masks.
# ─────────────────────────────────────────────────────────────────────────────
class ACTChunkHead(nn.Module):
    """Decode the full horizon in parallel. One learned query represents each future
    step. Queries attend to one another and to phase-conditioned memory before
    projection onto the four arm axes.
    """

    def __init__(self, hidden_dim: int = 3584, chunk_size: int = 5, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.nhead = 8
        self.head_dim = hidden_dim // self.nhead  # 448

        # Learn one DETR-style query per action step.
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)

        # Each block mixes action queries, attends to memory and applies a feed-forward layer.
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

        # Project features to the action coordinates.
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
        """Compute multi-head scaled dot-product attention. The optional [B, S] mask
        excludes padded keys before softmax. Recombine heads into a [B, T, D] output.
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
    ):
        """Return [B, K, 4] normalised action increments. Expand K learned queries per
        sample, prepend conditioning tokens to memory and extend the padding mask to
        match.
        """
        B, L, D = vlm_hidden.shape

        # Prepend the phase token to the decoder memory.
        phase_token_seq = phase_token.unsqueeze(1)  # [B, 1, D]
        memory = torch.cat([phase_token_seq, vlm_hidden], dim=1)  # [B, 1+L, D]

        # Extend the mask with a valid entry for the phase token.
        if memory_pad_mask is not None:
            phase_false = torch.zeros(B, 1, dtype=torch.bool, device=vlm_hidden.device)
            mem_pad = torch.cat([phase_false, memory_pad_mask], dim=1)  # [B, 1+L]
        else:
            mem_pad = torch.zeros(B, 1 + L, dtype=torch.bool, device=vlm_hidden.device)

        # Expand the learned action queries across the batch.
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Apply the stacked decoder blocks.
        for layer in self.layers:
            q = queries

            # The fixed-length query sequence has no padding.
            q_ln = layer["ln1"](q)
            attn_out = self._attn(layer["wq1"](q_ln), layer["wk1"](q_ln), layer["wv1"](q_ln))
            queries = q + layer["out1"](attn_out)

            # Each query attends to valid memory tokens.
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

        # Map each decoded query to the four arm-action coordinates.
        action_chunk = self.action_proj(queries)  # [B, K, 4]

        return action_chunk


# ─────────────────────────────────────────────────────────────────────────────
# MagVLAPolicy
# ─────────────────────────────────────────────────────────────────────────────
class MagVLAPolicy(nn.Module):
    """Combine the LoRA-adapted Qwen vision-language backbone, measured arm state and
    motion-aware phase classifier with phase-conditioned ACT for action prediction.
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

        # Load the pretrained vision-language backbone.
        model_config = Qwen2_5_VLConfig.from_pretrained(vlm_name)
        model_config.vision_config.hidden_size = 1280

        self.processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True)

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            config=model_config,
            torch_dtype=config.VLM_DTYPE,
            device_map=device,
            trust_remote_code=True,
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

        # Freeze backbone parameters outside the LoRA adapters.
        for name, param in self.vlm.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        # Construct the task-specific heads.
        self.state_proj = nn.Linear(4, hidden_dim)
        self.phase_head = PhaseHead(hidden_dim=hidden_dim, state_dim=4, num_phases=2)
        self.act_head = ACTChunkHead(hidden_dim=hidden_dim, chunk_size=chunk_size, num_layers=2)

        # Move the model components to the configured device.
        self.state_proj = self.state_proj.to(device, dtype=config.VLM_DTYPE)
        self.phase_head = self.phase_head.to(device, dtype=config.VLM_DTYPE)
        self.act_head = self.act_head.to(device, dtype=config.VLM_DTYPE)

        self.device = device

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize a single image to configured height and width when downsampling is
        enabled.
        """
        if self.image_downscale is not None:
            import cv2
            h, w = self.image_downscale
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def _run_vlm(self, images: torch.Tensor, task_texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode RGB history [B, T, H, W, 3] and one task string per sample. Return
        padded hidden states [B, L, D] and a Boolean [B, L] mask with True at padded
        positions.
        """
        B, T = images.shape[0], images.shape[1]

        hidden_states_list = []
        original_lengths = []
        for b in range(B):
            # Convert the observation history to PIL images for the processor.
            pil_images = []
            for t in range(T):
                img_np = images[b, t].cpu().numpy()
                img_np = self._resize_image(img_np)
                pil_img = Image.fromarray(img_np.astype(np.uint8))
                pil_images.append(pil_img)

            # Package the image history and task prompt for the VLM processor.
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
                "input_ids": inputs["input_ids"].to(self.device),
                "pixel_values": inputs["pixel_values"].to(self.device),
                "image_grid_thw": inputs["image_grid_thw"].to(self.device),
            }
            if "mm_token_type_ids" in inputs:
                vlm_inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(self.device)

            with torch.amp.autocast("cuda", dtype=config.VLM_DTYPE):
                outputs = self.vlm(**vlm_inputs, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states[-1]  # [1, L_b, D]
            hidden_states_list.append(hs)
            original_lengths.append(hs.shape[1])

        # Pad memory sequences to a common length and mark padded positions.
        if B == 1:
            return hidden_states_list[0], torch.zeros(
                (1, hidden_states_list[0].shape[1]),
                dtype=torch.bool,
                device=self.device,
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
        """Prepend the projected four-axis arm state to each memory sequence. Return [B,
        L+1, D] hidden states and a matching mask whose first entry is valid.
        """
        state_token = self.state_proj(state.to(dtype=self.state_proj.weight.dtype))  # [B, D]
        state_token = state_token.unsqueeze(1)  # [B, 1, D]
        vlm_hidden = torch.cat([state_token, vlm_hidden], dim=1)  # [B, 1+L, D]

        # Append False because the measured-state token is valid.
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
        """Map RGB history, task text and normalised arm states to phase logits and a
        five-step chunk. Each step has four coordinates ordered as left x, left y,
        right x, right y. For inference, use evaluation mode and omit ground-truth
        phase and action targets.
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

        # Supply the memory mask to ACT cross-attention.
        action_chunk = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
        )

        # Compute the supervised objectives when requested.
        if return_loss:
            if phase_gt is not None:
                phase_loss = F.cross_entropy(phase_logits, phase_gt)
            else:
                phase_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if action_delta_gt is not None:
                chunk_loss = F.smooth_l1_loss(action_chunk, action_delta_gt, beta=1.0)
            else:
                chunk_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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
        """Compute both supervised objectives from the supplied phase labels and action
        chunks. Return the combined loss, its components and the policy outputs.
        """
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

        # ── 4. ACT Chunk Head ──────────────────────────────────────────
        action_chunk = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
        )

        # Combine the action and phase losses.
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
# Policy and optimiser construction.
# ─────────────────────────────────────────────────────────────────────────────
def build_policy(device: str = config.DEVICE) -> MagVLAPolicy:
    """Construct the configured policy on the requested device."""
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
    """Return AdamW with separate learning-rate groups for adapters and task-specific
    layers, together with the warm-up, plateau and cosine scheduler.
    """
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

    # Warm up, hold the learning rate, then apply cosine decay to a positive floor.
    import math

    def lr_lambda(step):
        """Return the multiplier applied to each parameter group at the given
        optimisation step.
        """
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
