"""
MagVLA 配置 - Version 13 Flow Matching (基于 Qwen2.5-VL-7B，2-Phase，Flow Matching 小脑)

关键修改（相对 version13-IX-diffusion）：
- DiffusionChunkHead → FlowMatchingChunkHead（conditional flow matching，rectified flow）
- 不再依赖 diffusers（flow matching 无需 noise scheduler）
- chunk_loss 改为 velocity prediction MSE（与 diffusion 的 noise MSE 可直接比较）
- 连续时间 t ∈ [0,1]，直线路径 x_t = (1-t)*z + t*x_gt
- 推理：Euler 积分，FLOW_INFERENCE_STEPS 步（默认 10 步）
- RUN_NAME 更新为 v13-flow-matching
"""

import os
from pathlib import Path
import torch

# ── 数据集 ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("MARSS_DATA_DIR", "lerobot_datasetV2")).expanduser()

# ── 模型 ───────────────────────────────────────────────────────────────────
VLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_DTYPE = torch.bfloat16   # Qwen2.5-VL 官方推荐 bfloat16
IMAGE_H, IMAGE_W = 960, 1280  # 与数据集分辨率一致
IMAGE_HW = (IMAGE_H, IMAGE_W)

# 历史帧数（observation.images.main 窗口）
IMAGE_HISTORY = 4

# ACT chunk size（预测未来多少步）
CHUNK_SIZE = 5

# ── 图像下采样到更小分辨率加速训练（与 MagVLA 文档一致）
IMAGE_DOWNSCALE = (448, 448)  # H×W，448 是 Qwen2-VL 推荐倍数（14×32）

# ── 训练 (version13: True 4-GPU DDP, 2-Phase) ──────────────────────────────
BATCH_SIZE = 2              # per-GPU micro batch
ACCUM_STEPS = 8            # true 4GPU global batch = 2 * 4 * 8 = 64

# 学习率
LR = 1e-4                  # LoRA
LR_ACT_HEAD = 1.5e-4       # Flow Matching head + state_proj
LR_PHASE_HEAD = 1.5e-4     # Phase head（含 state_proj_phase + motion_proj）

WEIGHT_DECAY = 0.01

# scheduler
WARMUP_STEPS = 500
PLATEAU_STEPS = 1500
MIN_LR_RATIO = 0.1

MAX_STEPS = 5000
GRAD_CLIP = 1.0

# 损失权重
LAMBDA_PHASE = 0.5

# ── Flow Matching 配置 ────────────────────────────────────────────────────────
# Flow matching 训练时时间是连续的 t ∈ [0,1]，不需要 train_timesteps
# 推理时用 Euler 积分，步数越少越快（默认 10 步与 diffusion 保持可比）
FLOW_INFERENCE_STEPS = 10         # Euler 积分步数
FLOW_TIME_SCALE = 1000.0          # 给 sinusoidal time embed 的缩放（让 t∈[0,1] 变成 t'∈[0,1000]，避免低频问题）

# ── 日志 / 保存 ─────────────────────────────────────────────────────────────
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

# ── 设备 ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
