"""Experiment settings for ACT with an additional DistilBERT instruction token. DATA_DIR
points to recorded demonstrations. BATCH_SIZE counts samples per GPU microbatch and
ACCUM_STEPS sets the number of microbatches per optimiser update.
"""

import os
from pathlib import Path
import torch

# Dataset root.
DATA_DIR = Path(os.environ.get("MARSS_DATA_DIR", "lerobot_datasetV2")).expanduser()

# Backbone and input dimensions.
VLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_DTYPE = torch.bfloat16
IMAGE_H, IMAGE_W = 960, 1280
IMAGE_HW = (IMAGE_H, IMAGE_W)

IMAGE_HISTORY = 4
CHUNK_SIZE = 5

IMAGE_DOWNSCALE = (448, 448)

# Optimisation settings.
BATCH_SIZE = 2
ACCUM_STEPS = 8

LR = 1e-4
LR_ACT_HEAD = 1.5e-4
LR_PHASE_HEAD = 1.5e-4

WEIGHT_DECAY = 0.01

WARMUP_STEPS = 500
PLATEAU_STEPS = 1500
MIN_LR_RATIO = 0.1

MAX_STEPS = 5000
GRAD_CLIP = 1.0

LAMBDA_PHASE = 0.5

# Logging and checkpoint intervals.
LOG_INTERVAL = 10
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
PROJECT = "MagVLA"
RUN_NAME = "v13-text-injection"

# ── wandb ──────────────────────────────────────────────────────────────────
WANDB_KEY = os.environ.get("WANDB_API_KEY")
WANDB_PROJECT = "MagVLA"

# ── LoRA ───────────────────────────────────────────────────────────────────
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Device selection.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
