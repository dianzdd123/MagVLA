"""Mag-VLA implementation for DDIM diffusion. Action chunks describe planar increments
for the left and right magnet-mounted arms.
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

# Use the diffusers scheduler for training noise and DDIM sampling.
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
# DDIM action-chunk decoder.
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionChunkHead(nn.Module):
    """Learn a conditional noise predictor over an entire action chunk. The training
    target is the Gaussian noise added at a sampled timestep. At inference the DDIM
    scheduler iteratively denoises an initial noise sample.
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

        # Combine pooled VLM memory with the phase token.
        self.vlm_proj = nn.Linear(hidden_dim, cond_dim)
        self.phase_proj = nn.Linear(hidden_dim, cond_dim)

        # Project the sinusoidal time features with an MLP.
        # Compute sinusoidal features directly without trainable parameters.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition the denoiser on noisy actions, task features and time.
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
        # Initialise the denoiser's output prediction to zero.
        nn.init.zeros_(self.denoiser[-1].weight)
        nn.init.zeros_(self.denoiser[-1].bias)

        # ── DDIM noise scheduler ───────────────────────────────────────
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",   # Cosine noise schedule.
            clip_sample=False,                    # Use the supplied action increments without further clipping here.
            prediction_type="epsilon",            # Regress the noise added to the demonstration.
        )

        # Keep the attribute for compatibility with callers of the ACT interface.
        # Expose the phase embedding for callers using the common head interface.
        self.phase_embedding = None  # Reuse the phase token produced by the classifier.

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
        """Pool valid VLM tokens and combine their projected representation with the
        phase embedding to condition every sampling step.
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
        """Estimate noise in the flattened action chunk given the current noisy sample,
        timestep and observation conditioning.
        """
        # Project sinusoidal time features in the conditioning network's dtype.
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
        """With action_gt, return noise MSE and pass through the demonstrated chunk for
        logging. Omit action_gt to generate a new chunk using DDIM denoising. Only
        this sampling path returns actions suitable for comparison with
        demonstrations.
        """
        B = vlm_hidden.shape[0]
        device = vlm_hidden.device

        cond = self._get_cond(vlm_hidden, phase_token, memory_pad_mask)

        if action_gt is not None:
            # ════════════════════════════════════════════════════════════
            # Monitor the noise-prediction objective.
            # ════════════════════════════════════════════════════════════
            # Compute the noise schedule in float32.
            action_gt_fp32 = action_gt.float()                              # [B, K, A]

            # Sample a discrete noise-schedule timestep.
            timesteps = torch.randint(
                0, self.num_train_timesteps, (B,),
                device=device, dtype=torch.long,
            )

            # Sample Gaussian noise.
            noise = torch.randn_like(action_gt_fp32)                        # [B, K, A] fp32

            # Apply forward-process noise to the demonstrated action chunk.
            noisy_action = self.scheduler.add_noise(
                action_gt_fp32, noise, timesteps
            )                                                                # [B, K, A] fp32

            # Cast back to the conditioning dtype for the denoiser.
            noisy_flat = noisy_action.flatten(1).to(cond.dtype)             # [B, K*A]

            # Predict the added noise.
            noise_pred_flat = self._denoise_step(noisy_flat, cond, timesteps)
            noise_pred = noise_pred_flat.view(B, self.chunk_size, self.action_dim)

            # MSE loss in fp32
            loss = F.mse_loss(noise_pred.float(), noise)

            return {
                "chunk_loss": loss,
                "action_chunk": action_gt,  # Expose this value for training logs.
            }
        else:
            # ════════════════════════════════════════════════════════════
            # Generate an action chunk by DDIM sampling.
            # ════════════════════════════════════════════════════════════
            self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            # Initialise the reverse diffusion process with Gaussian noise.
            x = torch.randn(
                B, self.chunk_size, self.action_dim,
                device=device, dtype=torch.float32,
            )

            for t in self.scheduler.timesteps:
                t_batch = t.expand(B).to(device=device, dtype=torch.long)

                # Match the denoiser's conditioning dtype.
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
    """Combine the LoRA-adapted Qwen vision-language backbone, measured arm state and
    motion-aware phase classifier with DDIM diffusion for action prediction.
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

        # Freeze backbone parameters outside the LoRA adapters.
        for name, param in self.vlm.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        # Construct the task-specific heads.
        self.state_proj = nn.Linear(4, hidden_dim)
        self.phase_head = PhaseHead(hidden_dim=hidden_dim, state_dim=4, num_phases=2)
        # Keep the act_head attribute name for training and inference callers.
        # This variant assigns the diffusion decoder to that attribute.
        self.act_head = DiffusionChunkHead(
            hidden_dim=hidden_dim,
            action_dim=4,
            chunk_size=chunk_size,
            num_train_timesteps=config.DIFFUSION_TRAIN_TIMESTEPS,
            num_inference_steps=config.DIFFUSION_INFERENCE_STEPS,
        )

        # Move the model components to the configured device.
        self.state_proj = self.state_proj.to(device, dtype=config.VLM_DTYPE)
        self.phase_head = self.phase_head.to(device, dtype=config.VLM_DTYPE)
        self.act_head = self.act_head.to(device, dtype=config.VLM_DTYPE)

        self._device = device

    def _get_device(self) -> str:
        """Return the device of the backbone parameters."""
        return next(self.vlm.parameters()).device

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

        # Pad memory sequences to a common length and mark padded positions.
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

        # With targets, predict added noise. Without targets, sample with DDIM.
        chunk_out = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            action_gt=action_delta_gt,
        )
        action_chunk = chunk_out["action_chunk"]
        diffusion_chunk_loss = chunk_out["chunk_loss"]  # Noise-prediction MSE when a target chunk is supplied.

        # Compute the supervised objectives when requested.
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

        # ── 4. Diffusion Chunk Head ──────────────────────────────────────
        chunk_out = self.act_head(
            vlm_hidden,
            phase_token,
            memory_pad_mask=vlm_pad_mask,
            action_gt=action_delta_gt,
        )
        action_chunk = chunk_out["action_chunk"]
        diffusion_chunk_loss = chunk_out["chunk_loss"]  # noise prediction MSE

        # Combine the action and phase losses.
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
