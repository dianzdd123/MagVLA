"""
MagVLA 配置 - Version 13 (IX GPU Cloud 版，2-Phase + Direct Prompt Injection)

关键修改（相对 v13 原版）：
- PhaseHead num_phases 从 3 改为 2
- **新增直接 prompt 注入路径**：DistilBERT 编码 task_text → text_proj → text_token → 注入 ACT 小脑
- 动机：真机实验中 prompt 信息被 vision tokens 稀释，小脑无法区分任务
- 数据路径适配 IX 集群 /vol/bitbucket/$USER/ 路径
"""

import os
from pathlib import Path
import torch

# ── 数据集（IX 集群路径，由 SLURM $USER 自动填充）─────────────────────────
DATA_DIR = Path(os.environ.get("MARSS_DATA_DIR", "lerobot_datasetV2")).expanduser()

# ── 模型 ───────────────────────────────────────────────────────────────────
VLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_DTYPE = torch.bfloat16
IMAGE_H, IMAGE_W = 960, 1280
IMAGE_HW = (IMAGE_H, IMAGE_W)

IMAGE_HISTORY = 4
CHUNK_SIZE = 5

IMAGE_DOWNSCALE = (448, 448)

# ── 训练 ─────────────────────────────────────────────────────────────────────
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

# ── 日志 / 保存 ─────────────────────────────────────────────────────────────
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

# ── 设备 ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
