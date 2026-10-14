"""Bounded asynchronous retraining from trusted, labeled feedback."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from .detector import AdaptiveIDS


@dataclass
class RetrainingStats:
    batches_completed: int = 0
    samples_retrained: int = 0
    batches_dropped: int = 0
    last_error: str | None = None


class AsyncRetrainer:
    """Background worker; submit only independently confirmed labels.

    The queue is bounded to cap memory use. When full, the newest batch is
    rejected rather than silently deleting older feedback.
    """

    def __init__(
        self,
        model: AdaptiveIDS,
        batch_size: int = 64,
        queue_size: int = 8,
    ) -> None:
        if batch_size < 1 or queue_size < 1:
            raise ValueError("batch_size and queue_size must be positive.")
        self.model = model
        self.batch_size = batch_size
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="ids-retrainer", daemon=True)
        self.stats = RetrainingStats()
        self._worker.start()

    def submit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Queue labeled feedback; return False if the bounded queue is full."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if len(X) == 0:
            return True

        # Split large submissions into fixed-size batches.
        accepted = True
        for start in range(0, len(X), self.batch_size):
            item = (X[start:start + self.batch_size].copy(),
                    y[start:start + self.batch_size].copy())
            try:
                self._queue.put(item, timeout=0.05)
            except queue.Full:
                self.stats.batches_dropped += 1
                accepted = False
        return accepted

    def close(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            self._worker.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                X, y = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.model.partial_retrain(X, y)
                self.stats.batches_completed += 1
                self.stats.samples_retrained += len(y)
            except Exception as exc:  # Keep worker alive and expose the error.
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()
