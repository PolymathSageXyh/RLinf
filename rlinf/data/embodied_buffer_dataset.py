# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import queue
import threading
import time
from numbers import Real
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import concat_batch

logger = get_logger()


class ReplayBufferDataset(IterableDataset):
    """Dataset that samples batches from replay and demonstration buffers.

    This dataset provides an infinite iterator that yields batches sampled from
    a replay buffer and optionally a demonstration buffer. When both buffers are
    provided, ``demo_fraction`` controls the mixture and defaults to the legacy
    half-replay/half-demonstration behavior.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing offline demonstration trajectories
            and online human-in-the-loop trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        demo_fraction: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initializes the ReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            demo_fraction: Fraction of a mixed batch sampled from the demo
                buffer. Defaults to the legacy 0.5 split.
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size
        if (
            isinstance(demo_fraction, bool)
            or not isinstance(demo_fraction, Real)
            or not math.isfinite(demo_fraction)
            or not 0.0 <= demo_fraction <= 1.0
        ):
            raise ValueError("demo_fraction must be finite and in [0, 1].")
        self.demo_fraction = float(demo_fraction)

    def _sample_counts(self) -> tuple[int, int]:
        """Return replay/demo counts, preserving the legacy 0.5 split exactly."""
        if self.demo_buffer is None:
            return self.batch_size, 0
        if self.demo_fraction == 0.5:
            # Existing behavior intentionally yielded one fewer item for odd
            # batch sizes. Keep that behavior for old configurations.
            half_batch = self.batch_size // 2
            return half_batch, half_batch
        demo_count = int(self.batch_size * self.demo_fraction)
        return self.batch_size - demo_count, demo_count

    def _is_ready(self) -> bool:
        """Return whether every buffer used by the configured mixture is ready."""
        replay_count, demo_count = self._sample_counts()
        if replay_count and not self.replay_buffer.is_ready(
            self.min_replay_buffer_size
        ):
            return False
        if demo_count and not self.demo_buffer.is_ready(self.min_demo_buffer_size):
            return False
        return True

    def _sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample one batch using the configured replay/demo mixture."""
        replay_count, demo_count = self._sample_counts()
        replay_batch = self.replay_buffer.sample(replay_count) if replay_count else None
        if demo_count:
            assert self.demo_buffer is not None
            demo_batch = self.demo_buffer.sample(demo_count)
        else:
            demo_batch = None
        if replay_batch is None:
            if demo_batch is None:
                raise RuntimeError("Configured replay/demo mixture has zero samples.")
            return demo_batch
        if demo_batch is None:
            return replay_batch
        return concat_batch(replay_batch, demo_batch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an infinite iterator that yields batches.

        Waits until both buffers (if demo_buffer is provided) reach their
        minimum size requirements before yielding batches. When ready, samples
        from replay buffer only or from both replay and demo buffers.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        while True:
            if self._is_ready():
                yield self._sample_batch()

    def close(self) -> None:
        """Releases references to replay and demo buffers."""
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self) -> None:
        """Destructor that ensures buffers are cleaned up."""
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    """Dataset that prefetches batches from replay and demo buffers in background.

    This dataset extends ReplayBufferDataset by prefetching batches in a
    background thread, which can improve throughput by overlapping sampling
    with training. Batches are stored in a queue of configurable size.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing demonstration trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
        prefetch_size: Maximum number of batches to prefetch and store in queue.
        preload_queue: Queue holding prefetched batches.
        sample_thread: Background thread that samples batches.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        prefetch_size: int = 5,
        demo_fraction: float = 0.5,
    ) -> None:
        """Initializes the PreloadReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            prefetch_size: Maximum number of batches to prefetch and store in
                the queue. Defaults to 10.
            demo_fraction: Fraction of a mixed batch sampled from the demo
                buffer. Defaults to the legacy 0.5 split.
        """
        self._stop_event = threading.Event()
        self.sample_thread = None
        self._exception = None
        super().__init__(
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            batch_size=batch_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            demo_fraction=demo_fraction,
        )
        self.prefetch_size = prefetch_size
        assert self.prefetch_size > 0, f"{self.prefetch_size=} must be greater than 0"

        self.preload_queue = queue.Queue(maxsize=prefetch_size)

    def _sample_buffer(self) -> None:
        """Background thread target that continuously samples batches.

        Runs in a loop until stop event is set. Waits for buffers to be ready,
        samples batches, and puts them in the preload queue. If the queue is
        full, skips the sample and retries. Sleeps when buffers are not ready
        or when errors occur.
        """
        while not self._stop_event.is_set():
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            if self._is_ready():
                batch = self._sample_batch()
            else:
                time.sleep(3)
                continue

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                logger.info("Queue is full, skipping sample")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Error in ReplayBufferDataset: {e}")
                self._exception = e
                self._stop_event.set()
                break

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an iterator that yields prefetched batches.

        Starts the background sampling thread on first call. Retrieves batches
        from the preload queue and yields them. Stops when the stop event is set.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_buffer, daemon=True
            )
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    # Check if thread died with exception
                    if hasattr(self, "_exception"):
                        raise RuntimeError(
                            "Sampling thread failed"
                        ) from self._exception
                    break
                continue

    def close(self) -> None:
        """Stops the background sampling thread and cleans up resources.

        Sets the stop event and waits up to 10 seconds for the sampling thread
        to terminate. Logs a warning if the thread does not terminate in time.
        """
        self._stop_event.set()

        thread_timeout = 10
        if self.sample_thread is not None and self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )

    def __del__(self) -> None:
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader that returns the first batch element.

    Since the dataset already yields complete batches, this function simply
    extracts the batch from the list wrapper added by DataLoader.

    Args:
        batch: List containing a single batch dictionary.

    Returns:
        The unwrapped batch dictionary.
    """
    return batch[0]
