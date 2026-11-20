"""Construct episode-aligned training samples for ACT with an additional DistilBERT
instruction token from video, per-frame Parquet records and task metadata.
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
    """Seek to each requested video frame in order. Return BGR arrays, with None for
    individual decode failures. An unreadable video raises IOError.
    """
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
    """Index valid (episode, frame) pairs for one split. image_history controls how many
    observations end at the selected frame. chunk_size controls how many actions begin
    there. Optional normalisation reads statistics from meta/info.json.
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

        # Load split membership, task descriptions and dataset statistics.
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

        # Read the normalisation statistics supplied in dataset metadata.
        self.norm = self.info.get("normalization", None)

        # Index valid samples by episode and starting frame.
        # Exclude episode tails that cannot supply a complete target chunk.
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
                # Exclude frames labelled return (phase 2).
                phase = int(df["phase"].iloc[frame_idx])
                if phase == 2:
                    continue
                # Exclude windows that enter the return phase during the target chunk.
                chunk_phases = df["phase"].iloc[frame_idx : frame_idx + self.chunk_size].values
                if any(int(p) == 2 for p in chunk_phases):
                    continue
                self.samples.append((ep_idx, frame_idx))

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        """Count eligible starting frames after phase and horizon filtering."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return one aligned observation history, task instruction, current arm state,
        phase label and future action chunk. History indices clamp to frame zero at
        the start of an episode.
        """
        ep_idx, frame_idx = self.samples[idx]
        df = self.episode_data[ep_idx]
        N = len(df)

        # Current planar arm positions.
        state = np.array(df["observation.state"].iloc[frame_idx], dtype=np.float32)  # [4]

        # Ground-truth manipulation phase for this frame.
        phase_gt = int(df["phase"].iloc[frame_idx])  # 0 / 1 / 2

        # Look up the task instruction independently of the phase label.
        task_index = int(df["task_index"].iloc[frame_idx])
        task_text = self.tasks[task_index]

        # Gather the camera frames ending at the current time step.
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

        # Use the same frame indices for the arm-state history.
        state_history = np.stack([
            np.array(df["observation.state"].iloc[idx], dtype=np.float32)
            for idx in history_indices
        ])  # [4, 4]  (4 frames × 4 state dims)

        # Convert OpenCV BGR frames to RGB and resize.
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

        # Apply photometric augmentation only to training observations.
        if self.augment:
            import random

            # Randomly scale brightness by up to 15 percent.
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                images_aug = []
                for img in images:
                    img_f = img.astype(np.float32) * factor
                    images_aug.append(np.clip(img_f, 0, 255).astype(np.uint8))
                images = images_aug

            # Randomly scale contrast by up to 10 percent.
            if random.random() < 0.3:
                factor = random.uniform(0.90, 1.10)
                images_aug = []
                for img in images:
                    mean = img.mean()
                    img_f = (img.astype(np.float32) - mean) * factor + mean
                    images_aug.append(np.clip(img_f, 0, 255).astype(np.uint8))
                images = images_aug

        images = np.stack(images, axis=0)  # [4, H, W, 3] RGB

        # The sample index guarantees a complete future action window.
        action_delta_chunk = np.stack(
            df["action_delta"].iloc[frame_idx : frame_idx + self.chunk_size].values
        ).astype(np.float32)

        # Apply the stored state and action normalisation statistics.
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
        """Return the cached Parquet DataFrame for the requested episode index."""
        return self.episode_data[ep_idx]
