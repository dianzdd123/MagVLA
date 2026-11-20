"""Experiment settings for conditional flow matching. DATA_DIR points to recorded
demonstrations. BATCH_SIZE counts samples per GPU microbatch and ACCUM_STEPS sets the
number of microbatches per optimiser update.
"""

import os
from pathlib import Path
import torch

# Dataset root.
DATA_DIR = Path(os.environ.get("MARSS_DATA_DIR", "lerobot_datasetV2")).expanduser()

# Backbone and input dimensions.
VLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_DTYPE = torch.bfloat16   # Use bfloat16 for the vision-language backbone.
IMAGE_H, IMAGE_W = 960, 1280  # Height and width of the recorded frames.
IMAGE_HW = (IMAGE_H, IMAGE_W)

# Number of main-camera frames in the observation window.
IMAGE_HISTORY = 4

# Number of future action steps predicted per observation.
CHUNK_SIZE = 5

# Resize images for the vision-language backbone.
IMAGE_DOWNSCALE = (448, 448)  # Input height and width, both multiples of the visual patch size.

# Distributed training for the two manipulation phases.
BATCH_SIZE = 2              # per-GPU micro batch
ACCUM_STEPS = 8            # true 4GPU global batch = 2 * 4 * 8 = 64

# Learning rates for the parameter groups.
LR = 1e-4                  # LoRA
LR_ACT_HEAD = 1.5e-4       # Flow Matching head + state_proj
LR_PHASE_HEAD = 1.5e-4     # Phase classifier, including state and motion projections.

WEIGHT_DECAY = 0.01

# scheduler
WARMUP_STEPS = 500
PLATEAU_STEPS = 1500
MIN_LR_RATIO = 0.1

MAX_STEPS = 5000
GRAD_CLIP = 1.0

# Weight of the phase loss in the joint objective.
LAMBDA_PHASE = 0.5

# Flow-matching training and sampling settings.
# Draw continuous training times between zero and one.
# Control the number of Euler evaluations used to sample a chunk.
FLOW_INFERENCE_STEPS = 10         # Number of Euler integration steps.
FLOW_TIME_SCALE = 1000.0          # Rescale time before encoding it at multiple frequencies.

# Logging and checkpoint intervals.
LOG_INTERVAL = 10
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
PROJECT = "MagVLA"
RUN_NAME = "v13-flow-matching"

# ── wandb ──────────────────────────────────────────────────────────────────
WANDB_KEY = os.environ.get("WANDB_API_KEY")
WANDB_PROJECT = "MagVLA"

# ── LoRA ───────────────────────────────────────────────────────────────────
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# Device selection.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
