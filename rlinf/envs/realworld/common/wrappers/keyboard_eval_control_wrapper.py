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
"""Foot-pedal-gated wrapper for autonomous policy eval.

Pedal: ``a`` starts a rollout from idle; ``c`` records reward=1
("success"); ``b`` records reward=0 ("failure"). A recorded result is
returned as ``terminated=True`` at the end of the current action chunk.
"""

import copy
import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardEvalControlWrapper(gym.Wrapper):
    """Foot-pedal-gated start/stop for autonomous policy eval rollouts."""

    IDLE_POLL_S = 0.05
    PEDAL_DEBOUNCE_S = 0.2
    WAIT_HEARTBEAT_S = 10.0

    def __init__(self, env: gym.Env, action_chunk_size: int = 1):
        """Initialize the wrapper.

        Args:
            env: Environment that executes one primitive action per ``step``.
            action_chunk_size: Number of primitive actions in one policy chunk.
                A ``b`` or ``c`` result is latched until this boundary.

        Raises:
            ValueError: If ``action_chunk_size`` is not a positive integer.
        """
        super().__init__(env)
        if (
            isinstance(action_chunk_size, bool)
            or not isinstance(action_chunk_size, int)
            or action_chunk_size <= 0
        ):
            raise ValueError("action_chunk_size must be a positive integer")
        self.action_chunk_size = action_chunk_size
        self.listener = KeyboardListener()
        self._running = False
        self._last_obs: Any = None
        self._last_press_ts: dict[str, float] = {}
        self._chunk_step = 0
        self._pending_result: str | None = None

    def reset(self, *, seed=None, options=None):
        self._running = False
        self._chunk_step = 0
        self._pending_result = None
        self._last_press_ts.clear()
        self.listener.pop_pressed_keys()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = copy.deepcopy(obs)
        # Block until the operator has arranged the scene and presses 'a'.
        # This is intentional (the arms are homed and idle); log on entry and
        # emit a periodic heartbeat so the wait is not mistaken for a hang.
        self._log_info(
            "Arms homed and idle. Arrange the scene, then press pedal 'a' "
            "to start the next rollout (Ctrl-C to abort)."
        )
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info("Still waiting for pedal 'a' to start the rollout...")
            for key in self.listener.pop_pressed_keys():
                if key == "a":
                    self._running = True
                    self._log_info("Pedal 'a' pressed; starting rollout.")
                    return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Idle: poll pedal, hold robot via the controller's last target.
            time.sleep(self.IDLE_POLL_S)
            for key in self.listener.pop_pressed_keys():
                now = time.monotonic()
                if (
                    now - self._last_press_ts.get(key, -math.inf)
                    < self.PEDAL_DEBOUNCE_S
                ):
                    continue
                self._last_press_ts[key] = now
                if key == "a":
                    self._running = True
                    self._chunk_step = 0
                    self._pending_result = None
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        # Running: forward to the wrapped env.
        obs, reward, terminated, truncated, info = self.env.step(action)

        terminated = False
        truncated = False
        self._chunk_step += 1

        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now
            if key in ("b", "c") and self._pending_result is None:
                self._pending_result = "success" if key == "c" else "failure"
                self._log_info(
                    f"Pedal '{key}' pressed; finishing the current action chunk "
                    f"before recording {self._pending_result}."
                )
                break

        result: str | None = None
        if self._chunk_step == self.action_chunk_size:
            self._chunk_step = 0
            result = self._pending_result
            if result is not None:
                terminated = True
                reward = 1.0 if result == "success" else 0.0
                self._running = False
                self._pending_result = None
                # Outer observation wrappers mutate ``obs`` in place. Keep an
                # isolated raw copy for subsequent idle responses.
                self._last_obs = copy.deepcopy(obs)

        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _idle_response(self, event: str | None):
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return copy.deepcopy(self._last_obs), 0.0, False, False, info

    def _log_info(self, message: str) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
