# Copyright 2025 The RLinf Authors.
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

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class BaseKeyboardRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_mode: str = "always_replace"):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()
        self.reward_mode = reward_mode
        assert self.reward_mode in ["always_replace"]

    def _check_keypress(self) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        last_intervened, updated_reward, updated_terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
        return observation, reward, updated_terminated, truncated, info

    def reward_terminated(
        self,
    ) -> tuple[bool, float, bool]:
        last_intervened, terminated, keyboard_reward = self._check_keypress()
        return last_intervened, keyboard_reward, terminated


class KeyboardRewardDoneWrapper(BaseKeyboardRewardDoneWrapper):
    """Single-stage keyboard reward wrapper with optional recording gate.

    Without ``start_key`` this preserves the original behavior: ``a`` ends a
    failed episode, ``b`` emits a neutral reward, and ``c`` ends a successful
    episode.  When ``start_key`` is configured, reset enters a non-recording
    phase and the configured key starts a fresh recording at the current
    observation.  The phase metadata is consumed by ``CollectEpisode`` and the
    standalone real-world data collector.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_mode: str = "always_replace",
        start_key: str | None = None,
    ):
        super().__init__(env, reward_mode=reward_mode)
        if start_key is not None:
            start_key = str(start_key).lower()
            if len(start_key) != 1:
                raise ValueError(
                    f"keyboard_start_key must be one character, got {start_key!r}."
                )
            if start_key in {"a", "b", "c"}:
                raise ValueError(
                    "keyboard_start_key must not conflict with the single-stage "
                    f"reward keys a/b/c, got {start_key!r}."
                )
        self.start_key = start_key
        self._recording = start_key is None
        self._keyboard_event: str | None = None
        self._record_reset = False

    def reset(self, *, seed=None, options=None):
        self._recording = self.start_key is None
        self._keyboard_event = None
        self._record_reset = False
        if self.start_key is not None:
            # Do not let a press from the previous episode start the next one.
            self.listener.pop_pressed_keys()
            print(
                f"Keyboard collection is ready; press '{self.start_key}' "
                "to start recording."
            )
        return self.env.reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        self._keyboard_event = None
        self._record_reset = False

        if self.start_key is None:
            pressed_keys = [self.listener.get_key()]
        else:
            # Edge events avoid missing a quick tap between 10 Hz collection steps.
            pressed_keys = self.listener.pop_pressed_keys()

        for key in pressed_keys:
            if key is None:
                continue
            print(f"Key pressed: {key}")

            if not self._recording:
                if key == self.start_key:
                    self._recording = True
                    self._keyboard_event = "start"
                    self._record_reset = True
                    last_intervened = True
                # Reward/end keys are deliberately ignored before recording starts.
                break

            if key not in ["a", "b", "c"]:
                continue

            last_intervened = True
            if key == "a":
                reward = -1
                done = True
                self._keyboard_event = "end_failure"
            elif key == "b":
                reward = 0
                self._keyboard_event = "neutral"
            elif key == "c":
                reward = 1
                done = True
                self._keyboard_event = "end_success"
            break

        return last_intervened, done, reward

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, _terminated, truncated, info = self.env.step(action)
        last_intervened, keyboard_reward, terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = keyboard_reward

        if self.start_key is not None:
            info["pre_record"] = not self._recording
            info["record_reset"] = self._record_reset
            info["keyboard_phase"] = "rec" if self._recording else "pre"
            info["keyboard_event"] = self._keyboard_event
            info["segment_advance"] = False

        return observation, reward, terminated, truncated, info


class KeyboardRewardDoneMultiStageWrapper(BaseKeyboardRewardDoneWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stage_rewards = [0, 0.1, 1]

    def reset(self, *, seed=None, options=None):
        self.reward_stage = 0
        return super().reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key == "a":
            self.reward_stage = 0
        elif key == "b":
            self.reward_stage = 1
        elif key == "c":
            self.reward_stage = 2

        if self.reward_stage == 2:
            done = True

        reward = self.stage_rewards[self.reward_stage]
        if key == "q":
            reward = -1
            done = False
        return last_intervened, done, reward
