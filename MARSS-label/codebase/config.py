"""
MagVLA 配置 - Version 13 Label (基于 Qwen2.5-VL-7B backbone，带圆圈标记)

关键修改（相对 v13）：
- 在图像输入 VLM 之前，先在原图(1290×960)上标记两个圆圈：
  1. 红圆(object)：圆心 (object_x, object_y)
  2. 绿圆(target)：圆心 (target_x, target_y)
- 圆的半径从 annotations_point.csv 读取
- 标记完成后进行下采样，再输入 VLM
"""

import os
from pathlib import Path
import torch

# ── 数据集（IX 集群路径）─────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("MARSS_DATA_DIR", "lerobot_datasetV2")).expanduser()

# ── 标记点注释文件（用于在图像上标记圆圈）────────────────────────────────
# 格式: video_name, object_x, object_y, target_x, target_y, circle_radius
ANNOTATIONS_POINT_CSV = DATA_DIR / "annotations_point.csv"

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

# ── 训练 (version13-IX-label: 4-GPU DDP, 2-Phase + 圆圈标记) ──────────────
BATCH_SIZE = 2              # per-GPU micro batch
ACCUM_STEPS = 8            # true 4GPU global batch = 2 * 4 * 8 = 64

# 学习率
LR = 1e-4                  # LoRA
LR_ACT_HEAD = 1.5e-4       # ACT head + state_proj
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

# ── 日志 / 保存 ─────────────────────────────────────────────────────────────
LOG_INTERVAL = 10
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
PROJECT = "MagVLA"
RUN_NAME = "v13-qwenvl7b-label"

# ── wandb ───────────────────────────────────────────────────────────────────
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
