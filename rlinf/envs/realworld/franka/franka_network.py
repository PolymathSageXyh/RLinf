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

"""Host-network preparation for Franka's 1 kHz FCI connection."""

import os
import re
import shutil
import subprocess
from logging import Logger

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _find_executable(name: str) -> str | None:
    """Find a networking executable, including common administrator paths."""
    executable = shutil.which(name)
    if executable is not None:
        return executable

    for directory in ("/usr/sbin", "/sbin", "/usr/bin", "/bin"):
        candidate = os.path.join(directory, name)
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _command_error(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or f"exit code {result.returncode}").strip()


def _resolve_route_interface(robot_ip: str, ip_command: str) -> str:
    result = _run([ip_command, "route", "get", robot_ip])
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not resolve the network interface for Franka at {robot_ip}: "
            f"{_command_error(result)}"
        )

    match = re.search(r"(?:^|\s)dev\s+(\S+)", result.stdout)
    if match is None:
        raise RuntimeError(
            f"Could not parse the Franka network interface from: {result.stdout.strip()}"
        )
    return match.group(1)


def _is_usb_network_interface(interface: str) -> bool:
    """Return whether an interface is backed by a USB device in sysfs."""
    device_path = os.path.realpath(f"/sys/class/net/{interface}/device")
    return "/usb" in device_path


def _configure_low_latency_features(
    interface: str,
    ethtool: str,
    logger: Logger,
) -> None:
    """Best-effort removal of batching and flow-control latency on an FCI NIC."""
    settings = (
        (
            [ethtool, "-K", interface, "gro", "off", "gso", "off", "tso", "off"],
            "GRO/GSO/TSO offloads",
        ),
        (
            [ethtool, "-A", interface, "rx", "off", "tx", "off"],
            "Ethernet pause frames",
        ),
    )
    for command, description in settings:
        result = _run(command)
        error = _command_error(result)
        already_unchanged = "no pause parameters changed" in error.lower()
        if result.returncode != 0 and not already_unchanged:
            # Some dedicated NIC drivers do not expose every setting. EEE is the
            # mandatory check; these additional latency reductions are best effort.
            logger.warning(
                "Could not disable %s on Franka interface %s: %s",
                description,
                interface,
                error,
            )


def ensure_franka_network_ready(robot_ip: str, logger: Logger) -> str | None:
    """Prepare an Ethernet interface for Franka's 1 ms communication budget.

    The Franka Control Interface exchanges UDP packets at 1 kHz. Energy
    Efficient Ethernet (EEE) can add wake-up latency large enough to abort a
    motion with ``communication_constraints_violation``. This function finds
    the interface used to reach the robot, disables EEE when the NIC reports it
    as enabled, and turns off packet-batching offloads and pause frames when the
    driver supports those settings.

    Set ``RLINF_SKIP_FRANKA_NETWORK_TUNING=1`` only when network tuning is
    managed externally.

    Args:
        robot_ip: Franka Control IP address.
        logger: Logger used for diagnostics.

    Returns:
        The routed network-interface name, or ``None`` when the check is
        explicitly skipped or its required tools are unavailable.

    Raises:
        RuntimeError: If EEE is enabled but cannot be disabled and verified.
    """
    if os.environ.get("RLINF_SKIP_FRANKA_NETWORK_TUNING", "").lower() in _TRUTHY_VALUES:
        logger.warning(
            "Skipping Franka network tuning because "
            "RLINF_SKIP_FRANKA_NETWORK_TUNING is set."
        )
        return None

    ip_command = _find_executable("ip")
    if ip_command is None:
        logger.warning(
            "The 'ip' command is unavailable; RLinf cannot identify or tune the "
            "Franka network interface. Install iproute2 before robot operation."
        )
        return None

    interface = _resolve_route_interface(robot_ip, ip_command)
    if _is_usb_network_interface(interface):
        logger.warning(
            "Franka at %s is routed through USB Ethernet interface %s. USB bus and "
            "interrupt jitter can violate the 1 kHz FCI deadline even when ping shows "
            "no packet loss. Use a dedicated PCIe Ethernet interface for reliable "
            "robot control, especially when USB cameras are active.",
            robot_ip,
            interface,
        )
    ethtool = _find_executable("ethtool")
    if ethtool is None:
        logger.warning(
            "ethtool is unavailable; RLinf cannot verify EEE on Franka interface "
            "%s. Install ethtool and run 'sudo ethtool --set-eee %s eee off' "
            "before robot operation.",
            interface,
            interface,
        )
        return interface

    status = _run([ethtool, "--show-eee", interface])
    if status.returncode != 0:
        logger.warning(
            "Could not query EEE on Franka interface %s: %s",
            interface,
            _command_error(status),
        )
        return interface

    if re.search(r"EEE status:\s*enabled", status.stdout, re.IGNORECASE) is None:
        logger.info(
            "Franka network interface %s is routed to %s; EEE is disabled.",
            interface,
            robot_ip,
        )
    else:
        disable = _run([ethtool, "--set-eee", interface, "eee", "off"])
        if disable.returncode != 0:
            raise RuntimeError(
                f"EEE is enabled on Franka interface {interface}, but RLinf could not "
                f"disable it: {_command_error(disable)}. Run 'sudo ethtool --set-eee "
                f"{interface} eee off' on the robot-control host, or use a privileged "
                "host-network Docker container, before starting robot control."
            )

        verified = _run([ethtool, "--show-eee", interface])
        if verified.returncode != 0 or re.search(
            r"EEE status:\s*enabled", verified.stdout, re.IGNORECASE
        ):
            raise RuntimeError(
                f"RLinf requested EEE off on Franka interface {interface}, but could "
                "not verify that it was disabled. Do not start the 1 kHz control loop "
                "until 'ethtool --show-eee' reports 'EEE status: disabled'."
            )

        logger.warning(
            "Disabled EEE on Franka network interface %s to protect the 1 kHz FCI "
            "control loop. This setting may need to be reapplied after a host reboot.",
            interface,
        )

    _configure_low_latency_features(interface, ethtool, logger)
    return interface
