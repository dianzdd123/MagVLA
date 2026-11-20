"""
MagVLA Dataset - 带圆圈标记版本

关键修改（相对 v13）：
- 在图像输入 VLM 之前，先在原图(1290×960)上标记两个圆圈
- 红圆(object)：物体中心
- 绿圆(target)：目标点
- 标记完成后再进行下采样
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


def _load_annotations_point(annotations_path: Path) -> dict:
    """
    加载 annotations_point.csv，构建 episode_index -> 标记数据的映射。

    返回: {ep_idx: {"object_x": int, "object_y": int, "target_x": int,
                    "target_y": int, "circle_radius": int}}
    """
    if not annotations_path.exists():
        print(f"  [Warning] annotations_point.csv not found: {annotations_path}")
        return {}

    df = pd.read_csv(annotations_path)
    annotations = {}
    for _, row in df.iterrows():
        video_name = row["video_name"]
        ep_idx = int(video_name.replace("episode_", "").replace(".mp4", ""))
        annotations[ep_idx] = {
            "object_x": int(row["object_x"]),
            "object_y": int(row["object_y"]),
            "target_x": int(row["target_x"]),
            "target_y": int(row["target_y"]),
            "circle_radius": int(row["circle_radius"]),
        }
    return annotations


def _draw_circles_on_frame(
    frame: np.ndarray,
    object_x: int,
    object_y: int,
    target_x: int,
    target_y: int,
    circle_radius: int,
) -> np.ndarray:
    """
    在原图上标记两个圆圈：
    - 红圆(object)：物体中心，BGR=(255, 0, 0)
    - 绿圆(target)：目标点，BGR=(0, 255, 0)

    参数:
        frame: BGR 图像，原始分辨率 (1290×960)
        object_x, object_y: 红圆圆心坐标
        target_x, target_y: 绿圆圆心坐标
        circle_radius: 圆的半径

    返回:
        标记后的 BGR 图像
    """
    h, w = frame.shape[:2]

    object_x = max(0, min(object_x, w - 1))
    object_y = max(0, min(object_y, h - 1))
    target_x = max(0, min(target_x, w - 1))
    target_y = max(0, min(target_y, h - 1))

    cv2.circle(
        frame,
        center=(object_x, object_y),
        radius=circle_radius,
        color=(0, 0, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        center=(target_x, target_y),
        radius=circle_radius,
        color=(0, 255, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    return frame


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
        self.image_downscale = image_downscale
        self.augment = (split == "train")

        # ── 加载圆圈标记数据 ──────────────────────────────────────────────
        annotations_path = self.data_dir / "annotations_point.csv"
        self.annotations = _load_annotations_point(annotations_path)
        if self.annotations:
            print(f"  Loaded {len(self.annotations)} annotation entries from annotations_point.csv")

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
        self.samples = []
        self.episode_data = {}

        for ep_idx in self.episode_indices:
            parquet_path = (
                self.data_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
            )
            df = pd.read_parquet(str(parquet_path))
            self.episode_data[ep_idx] = df
            N = len(df)
            for frame_idx in range(N - self.chunk_size + 1):
                phase = int(df["phase"].iloc[frame_idx])
                if phase == 2:
                    continue
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
        state = np.array(df["observation.state"].iloc[frame_idx], dtype=np.float32)

        # ── 当前 phase ─────────────────────────────────────────────────
        phase_gt = int(df["phase"].iloc[frame_idx])

        # ── task_text ──────────────────────────────────────────────────
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
        ])

        # BGR -> RGB，并在下采样之前标记圆圈
        images = []
        for f in frames:
            if f is None:
                h, w = self.image_downscale or (960, 1280)
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                # ── 标记圆圈（在原图分辨率 1290×960 上标记）─────────────────
                if ep_idx in self.annotations:
                    ann = self.annotations[ep_idx]
                    f = _draw_circles_on_frame(
                        frame=f,
                        object_x=ann["object_x"],
                        object_y=ann["object_y"],
                        target_x=ann["target_x"],
                        target_y=ann["target_y"],
                        circle_radius=ann["circle_radius"],
                    )
                # ── BGR 转 RGB ────────────────────────────────────────────
                img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                # ── 下采样 ───────────────────────────────────────────────
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

            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                images_aug = []
                for img in images:
                    img_f = img.astype(np.float32) * factor
                    images_aug.append(np.clip(img_f, 0, 255).astype(np.uint8))
                images = images_aug

            if random.random() < 0.3:
                factor = random.uniform(0.90, 1.10)
                images_aug = []
                for img in images:
                    mean = img.mean()
                    img_f = (img.astype(np.float32) - mean) * factor + mean
                    images_aug.append(np.clip(img_f, 0, 255).astype(np.uint8))
                images = images_aug

        images = np.stack(images, axis=0)

        # ── 未来 chunk ─────────────────────────────────────────────────
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
            "observation.images.main": torch.from_numpy(images),
            "observation.state": torch.from_numpy(state),
            "state_history": torch.from_numpy(state_history),
            "task_text": task_text,
            "phase_gt": torch.tensor(phase_gt, dtype=torch.long),
            "action_delta_chunk": torch.from_numpy(action_delta_chunk),
            "chunk_size": self.chunk_size,
            "image_history": self.image_history,
        }

    def get_episode_info(self, ep_idx: int) -> dict:
        """返回指定 episode 的元信息"""
        return self.episode_data[ep_idx]
