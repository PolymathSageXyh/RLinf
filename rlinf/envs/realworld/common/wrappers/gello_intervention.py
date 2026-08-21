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

import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert
from rlinf.utils.logging import get_logger


class GelloIntervention(gym.ActionWrapper):
    """Action wrapper that overrides the policy action with GELLO teleop input.

    Args:
        env: The wrapped environment.
        port: Serial port of the GELLO device.  Must be provided
            (typically from the ``gello_port`` field in the env YAML config).
        gripper_enabled: Whether the gripper channel is present in the action
            space.  Determined by the ``no_gripper`` env config field.
        startup_timeout: Seconds to wait for the first valid GELLO frame.
        stale_timeout: Maximum age of a GELLO frame used for control.
        max_start_position_error: Maximum startup TCP position mismatch.
        max_start_orientation_error: Maximum startup TCP orientation mismatch.
    """

    def __init__(
        self,
        env,
        port: str,
        gripper_enabled: bool = True,
        startup_timeout: float = 10.0,
        stale_timeout: float = 1.0,
        max_start_position_error: float = 0.10,
        max_start_orientation_error: float = 0.50,
    ):
        super().__init__(env)

        self._logger = get_logger()
        self.gripper_enabled = gripper_enabled
        self.stale_timeout = stale_timeout
        self.expert = GelloExpert(port=port)
        self.expert.wait_until_ready(timeout=startup_timeout)
        target_pos, target_quat, _ = self.expert.get_action()
        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()
        position_error = float(np.linalg.norm(target_pos - tcp_pose[:3]))
        orientation_error = float(
            (R.from_quat(target_quat) * R.from_quat(tcp_pose[3:]).inv()).magnitude()
        )
        if (
            position_error > max_start_position_error
            or orientation_error > max_start_orientation_error
        ):
            raise RuntimeError(
                "GELLO and Franka are not aligned at startup; refusing to enable "
                "absolute-pose teleoperation. "
                f"GELLO pose={np.concatenate((target_pos, target_quat))}, "
                f"Franka pose={tcp_pose}, position_error={position_error:.3f}m "
                f"(limit={max_start_position_error:.3f}m), "
                f"orientation_error={orientation_error:.3f}rad "
                f"(limit={max_start_orientation_error:.3f}rad). Recalibrate the "
                "GELLO joint offsets/signs and place both arms in matching poses."
            )
        self._logger.info(
            "GELLO is ready and aligned on %s: position_error=%.3fm, "
            "orientation_error=%.3frad",
            port,
            position_error,
            orientation_error,
        )
        self.last_intervene = 0
        self._logged_first_action = False

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.expert.ready:
            raise RuntimeError("GELLO reader is no longer ready.")
        update_age = self.expert.last_update_age
        if update_age > self.stale_timeout:
            raise RuntimeError(
                "GELLO input became stale: last valid frame was "
                f"{update_age:.3f}s ago (limit={self.stale_timeout:.3f}s)."
            )

        target_pos, target_quat, target_gripper = self.expert.get_action()
        r_target = R.from_quat(target_quat.copy())
        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()

        tcp_pos = tcp_pose[:3]
        tcp_quat = tcp_pose[3:]
        r_tcp = R.from_quat(tcp_quat.copy())

        r_delta = r_target * r_tcp.inv()
        delta_euler = r_delta.as_euler("xyz")

        delta_pos = target_pos - tcp_pos

        action_scale = self.get_wrapper_attr("get_action_scale")()
        delta_pos = delta_pos / action_scale[0]
        delta_euler = delta_euler / action_scale[1]

        expert_a = np.concatenate((delta_pos, delta_euler), axis=0)
        expert_a = np.clip(expert_a, -1.0, 1.0)

        gripper_active = False
        if self.gripper_enabled:
            target_gripper = target_gripper / action_scale[2]
            target_gripper = -(2 * target_gripper - 1.0)
            target_gripper = np.clip(target_gripper, -1.0, 1.0)
            gripper_active = np.abs(target_gripper).item() > 0.5
            expert_a = np.concatenate((expert_a, target_gripper), axis=0)

        if not self._logged_first_action:
            self._logger.info(
                "GELLO first control action: tcp_pos=%s, target_pos=%s, action=%s",
                np.array2string(tcp_pos, precision=3),
                np.array2string(target_pos, precision=3),
                np.array2string(expert_a, precision=3),
            )
            self._logged_first_action = True

        if np.linalg.norm(expert_a[:6]) > 0.001 or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action

        return obs, rew, done, truncated, info
