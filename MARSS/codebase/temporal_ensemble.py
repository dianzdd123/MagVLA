"""Offline implementation of Mag-VLA Eq. (2), for a single episode."""
from collections import deque

import numpy as np


class TemporalEnsemble:
    """Create one instance per episode. Call update once per replanning step."""

    def __init__(self):
        self._chunks = deque(maxlen=5)
        self._last_step = None

    def update(self, step, chunk):
        """Combine predictions for this step in a single episode. Supply a strictly
        increasing integer step and a finite [5, 4] chunk. Each active chunk
        contributes its row at the current age, weighted by exp(-0.01 * age). Return
        four actuator increments in the input units.
        """
        if not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("step must increase within an episode")
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape != (5, 4) or not np.isfinite(chunk).all():
            raise ValueError("expected five finite four-axis actions")
        while self._chunks and step - self._chunks[0][0] >= 5:
            self._chunks.popleft()
        self._chunks.append((step, chunk.copy()))
        self._last_step = step
        ages = [step - start for start, _ in self._chunks]
        actions = [actions[step - start] for start, actions in self._chunks]
        return np.average(actions, axis=0, weights=np.exp(-0.01 * np.asarray(ages)))
