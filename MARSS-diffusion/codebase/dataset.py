"""
MagVLA Dataset

读取 lerobot_datasetV2，按 split (train/val/test) 构造训练样本。
每帧构造:
  - 4 帧历史图像 (observation.images.main)
  - 1 个当前 state (observation.state)
  - 1 个语言指令 (task_text)
  - 1 个 phase_gt
  - 未来 5 步的 action_delta chunk (ACT target)
"""

import json
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Tuple

__all__ = ["MagVLADataset"]


def _read_video_frames(video_path: str, indices: list) -> list:
    """从 mp4 读取指定帧索引列表，返回 BGR 图像列表 (H, W, 3)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame if ret else None)

    cap.release()
    return frames


class MagVLADataset(Dataset):
    """
    参数:
        data_dir: lerobot_datasetV2 根目录
        split: "train" / "val" / "test"
        image_history: 历史图像帧数，默认 4
        chunk_size: ACT 预测的未来步数，默认 5
        normalize: 是否对 state/action_delta 做归一化，默认 True
        image_downscale: 图像下采样分辨率 (H, W)，默认 None（使用原分辨率）
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_history: int = 4,
        chunk_size: int = 5,
        normalize: bool = True,
        image_downscale: Optional[Tuple[int, int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_history = image_history
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.image_downscale = image_downscale  # (H, W)
        self.augment = (split == "train")

        # ── 读取 meta ──────────────────────────────────────────────────────
        with open(self.data_dir / "meta" / "splits.json", encoding="utf-8") as f:
            splits = json.load(f)
        self.episode_indices = splits[split]

        with open(self.data_dir / "meta" / "tasks.jsonl", encoding="utf-8") as f:
            self.tasks = {}
            for line in f:
                item = json.loads(line)
                self.tasks[item["task_index"]] = item["task"]

        with open(self.data_dir / "meta" / "info.json", encoding="utf-8") as f:
            self.info = json.load(f)

        # 归一化参数（仅 train split）
        self.norm = self.info.get("normalization", None)

        # ── 建立 (episode_index, frame_index) 索引 ─────────────────────────
        # 过滤掉 episode 末尾不足 chunk_size 的帧（避免未来 chunk 全是重复的最后一个 action）
        self.samples = []
        self.episode_data = {}  # ep_idx -> parquet DataFrame

        for ep_idx in self.episode_indices:
            parquet_path = (
                self.data_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
            )
            df = pd.read_parquet(str(parquet_path))
            self.episode_data[ep_idx] = df
            N = len(df)
            for frame_idx in range(N - self.chunk_size + 1):
                # 跳过 return 阶段（phase==2）的帧
                phase = int(df["phase"].iloc[frame_idx])
                if phase == 2:
                    continue
                # 跳过 chunk 跨越到 return 阶段的帧（避免 action target 包含回家动作）
                chunk_phases = df["phase"].iloc[frame_idx : frame_idx + self.chunk_size].values
                if any(int(p) == 2 for p in chunk_phases):
                    continue
                self.samples.append((ep_idx, frame_idx))

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx = self.samples[idx]
        df = self.episode_data[ep_idx]
        N = len(df)

        # ── 当前状态 ────────────────────────────────────────────────────
        state = np.array(df["observation.state"].iloc[frame_idx], dtype=np.float32)  # [4]

        # ── 当前 phase ─────────────────────────────────────────────────
        phase_gt = int(df["phase"].iloc[frame_idx])  # 0 / 1 / 2

        # ── task_text（纯任务描述，不泄漏 phase 信息）────────────────────
        task_index = int(df["task_index"].iloc[frame_idx])
        task_text = self.tasks[task_index]

        # ── 历史图像 ───────────────────────────────────────────────────
        video_path = str(
            self.data_dir
            / "videos"
            / "chunk-000"
            / "observation.images.main"
            / f"episode_{ep_idx:06d}.mp4"
        )
        history_indices = [
            max(0, frame_idx - (self.image_history - 1 - i))
            for i in range(self.image_history)
        ]
        frames = _read_video_frames(video_path, history_indices)

        # ── 历史状态（与图像帧对齐）──────────────────────────────────
        state_history = np.stack([
            np.array(df["observation.state"].iloc[idx], dtype=np.float32)
            for idx in history_indices
        ])  # [4, 4]  (4 frames × 4 state dims)

        # BGR -> RGB，并 resize
        images = []
        for f in frames:
            if f is None:
                h, w = self.image_downscale or (960, 1280)
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                if self.image_downscale:
                    img = cv2.resize(
                        img,
                        (self.image_downscale[1], self.image_downscale[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
            images.append(img)

        # ── 轻量图像增强（仅 train）───────────────────────────────
        if self.augment:
            import random

            # 亮度增强：±15%
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                images_aug = []
                for img in images:
                    img_f = img.astype(np.float32) * factor
                    images_aug.append(np.clip(img_f, 0, 255).astype(np.uint8))
                images = images_aug

            # 对比度增强：±10%
            if random.random() < 0.3:
                factor = random.uniform(0.90, 1.10)
                images_aug = []
                for img in images:
                    mean = img.mean()
                    img_f = (img.astype(np.float32) - mean) * factor + mean
                    images_aug.append(np.clip(img_f, 0, 255).astype(np.uint8))
                images = images_aug

        images = np.stack(images, axis=0)  # [4, H, W, 3] RGB

        # ── 未来 chunk（frame_idx 已保证 <= N - chunk_size，无需 min 截断）───
        action_delta_chunk = np.stack(
            df["action_delta"].iloc[frame_idx : frame_idx + self.chunk_size].values
        ).astype(np.float32)

        # ── 归一化 ─────────────────────────────────────────────────────
        if self.normalize and self.norm:
            dm = np.array(self.norm["action_delta_mean"], dtype=np.float32)
            ds = np.array(self.norm["action_delta_std"], dtype=np.float32)
            sm = np.array(self.norm["state_mean"], dtype=np.float32)
            ss = np.array(self.norm["state_std"], dtype=np.float32)
            state = (state - sm) / (ss + 1e-8)
            state_history = (state_history - sm) / (ss + 1e-8)
            action_delta_chunk = (action_delta_chunk - dm) / (ds + 1e-8)

        return {
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "observation.images.main": torch.from_numpy(images),  # [4, H, W, 3]
            "observation.state": torch.from_numpy(state),        # [4]
            "state_history": torch.from_numpy(state_history),    # [4, 4]
            "task_text": task_text,
            "phase_gt": torch.tensor(phase_gt, dtype=torch.long),
            "action_delta_chunk": torch.from_numpy(action_delta_chunk),  # [5, 4]
            "chunk_size": self.chunk_size,
            "image_history": self.image_history,
        }

    def get_episode_info(self, ep_idx: int) -> dict:
        """返回指定 episode 的元信息"""
        return self.episode_data[ep_idx]
