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

"""Small, PyTorch-only helpers for pretrained SACFlow fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Mapping

import torch
import torch.nn.functional as F


class SACFlowFinetunePhase(str, Enum):
    """Optimization phase selected by online transition count."""

    WARMUP = "warmup"
    ONLINE = "online"


class SACFlowFinetuneController:
    """Track the transition clock and update-to-data-ratio budget.

    The controller deliberately has no worker or distributed dependencies, so
    synchronous and asynchronous workers use exactly the same phase semantics.
    Demonstration samples do not advance this clock; the worker records only
    transitions newly received from the online environment.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        *,
        warmup_transitions: int,
        updates_per_transition: float,
    ) -> None:
        if (
            isinstance(warmup_transitions, bool)
            or not isinstance(warmup_transitions, Integral)
            or warmup_transitions < 0
        ):
            raise ValueError("warmup_transitions must be a non-negative integer.")
        if (
            isinstance(updates_per_transition, bool)
            or not isinstance(updates_per_transition, Real)
            or not math.isfinite(float(updates_per_transition))
            or updates_per_transition < 0
        ):
            raise ValueError("updates_per_transition must be finite and non-negative.")

        self.warmup_transitions = int(warmup_transitions)
        self.updates_per_transition = float(updates_per_transition)
        self.online_transitions = 0
        self.update_credit = 0.0
        self.completed_updates = 0

    @property
    def phase(self) -> SACFlowFinetunePhase:
        """Return the phase selected by the online transition clock."""
        if self.online_transitions < self.warmup_transitions:
            return SACFlowFinetunePhase.WARMUP
        return SACFlowFinetunePhase.ONLINE

    def _phase_for_update_index(self, update_index: int) -> SACFlowFinetunePhase:
        if self.updates_per_transition == 0:
            return self.phase
        virtual_transition = (update_index + 1) / self.updates_per_transition
        if virtual_transition <= self.warmup_transitions:
            return SACFlowFinetunePhase.WARMUP
        return SACFlowFinetunePhase.ONLINE

    @property
    def next_update_phase(self) -> SACFlowFinetunePhase:
        """Return the phase of the next optimizer update in the UTD queue."""
        return self._phase_for_update_index(self.completed_updates)

    @property
    def available_updates(self) -> int:
        """Return the number of whole optimizer updates currently budgeted."""
        # The epsilon prevents values such as 0.9999999999999999 from delaying
        # an update solely because of floating-point accumulation.
        return max(0, int(math.floor(self.update_credit + 1e-12)))

    def record_online_transitions(self, count: int) -> None:
        """Add newly collected environment transitions to the schedule."""
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("online transition count must be a non-negative integer.")
        self.online_transitions += int(count)
        self.update_credit += int(count) * self.updates_per_transition

    def claim_updates(self, max_updates: int | None = None) -> int:
        """Consume and return whole updates from the current UTD budget."""
        return len(self.claim_update_phases(max_updates=max_updates))

    def claim_update_phases(
        self, max_updates: int | None = None
    ) -> tuple[SACFlowFinetunePhase, ...]:
        """Consume updates and retain warm-up work across batched receives.

        Phase assignment uses the virtual transition completion represented by
        each optimizer update (``(update_index + 1) / UTD``), rather than the
        latest receive counter.
        Consequently, receiving a chunk that crosses ``warmup_transitions`` does
        not incorrectly label every queued update as online.
        """
        if max_updates is not None:
            if (
                isinstance(max_updates, bool)
                or not isinstance(max_updates, Integral)
                or max_updates <= 0
            ):
                raise ValueError("max_updates must be a positive integer or None.")
        count = self.available_updates
        if max_updates is not None:
            count = min(count, int(max_updates))
        phases: list[SACFlowFinetunePhase] = []
        for offset in range(count):
            update_index = self.completed_updates + offset
            phases.append(self._phase_for_update_index(update_index))
        self.update_credit = max(0.0, self.update_credit - count)
        self.completed_updates += count
        return tuple(phases)

    def state_dict(self) -> dict[str, int | float | str]:
        """Serialize phase and UTD state for exact resume."""
        return {
            "version": self._STATE_VERSION,
            "warmup_transitions": self.warmup_transitions,
            "updates_per_transition": self.updates_per_transition,
            "online_transitions": self.online_transitions,
            "update_credit": self.update_credit,
            "completed_updates": self.completed_updates,
            "phase": self.next_update_phase.value,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore state while rejecting schedule-changing config drift."""
        version = int(state_dict.get("version", 0))
        if version != self._STATE_VERSION:
            raise ValueError(
                f"Unsupported SACFlow fine-tune state version {version}; "
                f"expected {self._STATE_VERSION}."
            )
        saved_warmup = int(state_dict["warmup_transitions"])
        saved_utd = float(state_dict["updates_per_transition"])
        if saved_warmup != self.warmup_transitions or not math.isclose(
            saved_utd, self.updates_per_transition
        ):
            raise ValueError(
                "SACFlow fine-tune schedule differs from the checkpoint: "
                f"saved warmup/UTD=({saved_warmup}, {saved_utd}), current="
                f"({self.warmup_transitions}, {self.updates_per_transition})."
            )

        online_transitions = int(state_dict["online_transitions"])
        update_credit = float(state_dict["update_credit"])
        completed_updates = int(state_dict["completed_updates"])
        if online_transitions < 0 or completed_updates < 0:
            raise ValueError(
                "Checkpoint transition/update counters must be non-negative."
            )
        if not math.isfinite(update_credit) or update_credit < -1e-9:
            raise ValueError(
                "Checkpoint update_credit must be finite and non-negative."
            )

        self.online_transitions = online_transitions
        self.update_credit = max(0.0, update_credit)
        self.completed_updates = completed_updates
        saved_phase = state_dict.get("phase", None)
        if saved_phase is not None and saved_phase != self.next_update_phase.value:
            raise ValueError(
                "Checkpoint SACFlow phase is inconsistent with its counters: "
                f"saved={saved_phase!r}, derived={self.next_update_phase.value!r}."
            )


@dataclass(frozen=True)
class SACFlowActorLoss:
    """Decomposed actor loss used for logging and focused unit tests."""

    total: torch.Tensor
    sac: torch.Tensor
    behavior: torch.Tensor


def compute_frozen_anchor_actor_loss(
    *,
    policy_actions: torch.Tensor,
    anchor_actions: torch.Tensor,
    behavior_beta: float,
    phase: SACFlowFinetunePhase,
    joint_log_prob: torch.Tensor | None = None,
    q_value: torch.Tensor | None = None,
    alpha: float | torch.Tensor = 0.0,
) -> SACFlowActorLoss:
    """Compose warm-up anchor-only or online SAC-plus-anchor actor loss."""
    if not math.isfinite(behavior_beta) or behavior_beta < 0:
        raise ValueError("behavior_beta must be finite and non-negative.")
    if policy_actions.shape != anchor_actions.shape:
        raise ValueError(
            "policy_actions and anchor_actions must have the same shape, got "
            f"{tuple(policy_actions.shape)} and {tuple(anchor_actions.shape)}."
        )

    anchor_actions = anchor_actions.to(
        device=policy_actions.device, dtype=policy_actions.dtype
    )
    behavior_loss = F.mse_loss(policy_actions, anchor_actions.detach())
    # Keep the zero term attached to the actor graph in warm-up, including when
    # behavior_beta is zero, so the worker can use one backward path.
    sac_loss = policy_actions.sum() * 0.0
    if phase is SACFlowFinetunePhase.ONLINE:
        if joint_log_prob is None or q_value is None:
            raise ValueError(
                "joint_log_prob and q_value are required in the online phase."
            )
        actor_alpha = alpha.detach() if isinstance(alpha, torch.Tensor) else alpha
        sac_loss = ((actor_alpha * joint_log_prob) - q_value).mean()

    return SACFlowActorLoss(
        total=sac_loss + behavior_beta * behavior_loss,
        sac=sac_loss,
        behavior=behavior_loss,
    )


def make_common_flow_noise(
    *,
    batch_size: int,
    action_dim: int,
    denoising_steps: int,
    device: torch.device | str | int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    """Create one immutable-by-convention noise path for policy and anchor."""
    for name, value in (
        ("batch_size", batch_size),
        ("action_dim", action_dim),
        ("denoising_steps", denoising_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    shape = (int(batch_size), int(action_dim))
    initial_noise = torch.randn(shape, device=device, dtype=dtype)
    step_noises = tuple(
        torch.randn(shape, device=device, dtype=dtype)
        for _ in range(int(denoising_steps))
    )
    return {"initial_noise": initial_noise, "step_noises": step_noises}


__all__ = [
    "SACFlowActorLoss",
    "SACFlowFinetuneController",
    "SACFlowFinetunePhase",
    "compute_frozen_anchor_actor_loss",
    "make_common_flow_noise",
]
