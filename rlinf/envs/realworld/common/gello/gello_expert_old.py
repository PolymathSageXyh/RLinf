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

import threading
import time

import numpy as np

from rlinf.utils.logging import get_logger


class GelloExpert:
    """Interface to the GELLO teleoperation device.

    Continuously reads GELLO joint positions in a background thread,
    computes the corresponding TCP pose via forward kinematics, and
    exposes the result through :meth:`get_action`.

    Args:
        port: Serial port of the GELLO device, e.g.
            ``"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"``.
    """

    def __init__(self, port: str):
        from gello_teleop.franka_fk import FrankaFK
        from gello_teleop.gello_teleop_agent import GelloTeleopAgent

        self._logger = get_logger()
        self.port = port
        self.agent = GelloTeleopAgent(port=port)
        self.fk = FrankaFK()

        self.state_lock = threading.Lock()
        self._ready = False
        self._ready_event = threading.Event()
        self._error: Exception | None = None
        self._last_update_time: float | None = None
        self.latest_data = {
            "target_pos": np.zeros(3),
            "target_quat": np.zeros(4),
            "gripper": np.zeros(1),
        }
        self.thread = threading.Thread(
            target=self._read_gello,
            daemon=True,
            name="gello-reader",
        )
        self.thread.start()

    def _read_gello(self):
        try:
            while True:
                gello_joints, gello_gripper = self.agent.get_action()
                gello_joints = np.asarray(gello_joints, dtype=np.float64)
                if gello_joints.shape != (7,) or not np.all(np.isfinite(gello_joints)):
                    raise ValueError(
                        "GELLO must return seven finite arm joints; "
                        f"received shape={gello_joints.shape}, values={gello_joints}."
                    )

                target_pos, target_quat = self.fk.get_fk(gello_joints)
                target_pos = np.asarray(target_pos, dtype=np.float64)
                target_quat = np.asarray(target_quat, dtype=np.float64)
                gello_gripper = np.asarray([gello_gripper], dtype=np.float64)
                if (
                    target_pos.shape != (3,)
                    or target_quat.shape != (4,)
                    or not np.all(np.isfinite(target_pos))
                    or not np.all(np.isfinite(target_quat))
                    or np.linalg.norm(target_quat) < 1e-6
                ):
                    raise ValueError(
                        "GELLO forward kinematics returned an invalid pose: "
                        f"position={target_pos}, quaternion={target_quat}."
                    )

                first_frame = False
                with self.state_lock:
                    self.latest_data["target_pos"] = target_pos.copy()
                    self.latest_data["target_quat"] = target_quat.copy()
                    self.latest_data["gripper"] = gello_gripper.copy()
                    self._last_update_time = time.monotonic()
                    first_frame = not self._ready
                    self._ready = True
                    self._ready_event.set()

                if first_frame:
                    self._logger.info(
                        "GELLO first frame received on %s: joints=%s, "
                        "target_pos=%s, target_quat=%s, gripper=%s",
                        self.port,
                        np.array2string(gello_joints, precision=3),
                        np.array2string(target_pos, precision=3),
                        np.array2string(target_quat, precision=3),
                        np.array2string(gello_gripper, precision=3),
                    )

                time.sleep(0.001)
        except Exception as exc:
            with self.state_lock:
                self._error = exc
                self._ready = False
                self._ready_event.set()
            self._logger.exception("GELLO reader stopped on %s", self.port)

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        """Wait for the first valid device frame or raise a useful error."""
        if not self._ready_event.wait(timeout=timeout):
            raise TimeoutError(
                f"No valid GELLO frame was received from {self.port!r} within "
                f"{timeout:.1f}s. Opening the serial port alone is not sufficient. "
                "Check GELLO power/cable, Dynamixel IDs 1-8, baud rate, exclusive "
                "port access, and that the runtime imports the edited gello package."
            )
        with self.state_lock:
            error = self._error
            ready = self._ready
        if error is not None:
            raise RuntimeError(
                f"GELLO reader failed on serial port {self.port!r}: {error}"
            ) from error
        if not ready:
            raise RuntimeError(
                f"GELLO reader on serial port {self.port!r} stopped before its first frame."
            )

    @property
    def ready(self) -> bool:
        """Whether at least one GELLO frame has been received."""
        with self.state_lock:
            return self._ready and self._error is None

    @property
    def last_update_age(self) -> float:
        """Seconds elapsed since the most recent valid GELLO frame."""
        with self.state_lock:
            last_update_time = self._last_update_time
        if last_update_time is None:
            return float("inf")
        return time.monotonic() - last_update_time

    def get_action(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(target_pos, target_quat, gripper)`` from the latest GELLO reading."""
        with self.state_lock:
            if self._error is not None:
                raise RuntimeError(
                    f"GELLO reader failed on serial port {self.port!r}: {self._error}"
                ) from self._error
            return (
                self.latest_data["target_pos"].copy(),
                self.latest_data["target_quat"].copy(),
                self.latest_data["gripper"].copy(),
            )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test the GELLO expert.")
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port of the GELLO device.",
    )
    args = parser.parse_args()

    gello = GelloExpert(port=args.port)
    gello.wait_until_ready()
    with np.printoptions(precision=3, suppress=True):
        while True:
            target_pos, target_quat, gripper = gello.get_action()
            #gello_joints, gello_gripper = gello.agent.get_action()
            print(
                f"pos={target_pos}  quat={target_quat}  gripper={gripper}",
                #f"gello_joints={gello_joints}",
                end="\r",
            )
            time.sleep(0.1)
