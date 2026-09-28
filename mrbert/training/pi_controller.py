# Copyright 2026 Aronima Dass, Alina Tianhui Huang, Hiva Mohammadzadeh.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PI controller for targeting a specific token deletion rate (paper Section 3.2, Eq. 4-6).

Dynamically adjusts the deletion loss coefficient α each training step so that the
observed deletion rate tracks a target rate δ.

Control law:
    error  = target_rate - actual_rate
    p_acc  = gamma * p_acc + (1 - gamma) * kp * error   # EMA of proportional term
    i_acc  = i_acc + ki * error                          # integral of error
    alpha  = max(0, p_acc + i_acc)

Reference: MrT5 paper Section 3.2 and trainer.py lines 262-266.
"""


class PIController:
    """
    Proportional-Integral controller that adjusts deletion loss weight α.

    Args:
        target_rate: Target fraction of tokens to delete (δ in the paper, e.g. 0.3).
        kp: Proportional gain. Higher values → faster convergence but more oscillation.
        ki: Integral gain. Drives steady-state error to zero over time.
        gamma: EMA smoothing factor for the proportional term (default 0.9).
        alpha_0: Initial value of α (default 0.0 — let the controller ramp up from zero).
    """

    def __init__(
        self,
        target_rate: float,
        kp: float = 0.5,
        ki: float = 1e-5,
        gamma: float = 0.9,
        alpha_0: float = 0.0,
    ):
        self.target_rate = target_rate
        self.kp = kp
        self.ki = ki
        self.gamma = gamma
        self.p_acc = 0.0   # EMA proportional accumulator
        self.i_acc = 0.0   # Integral accumulator

    def update(self, actual_rate: float, return_state: bool = False):
        """
        Compute the updated deletion loss coefficient α for this step.

        Args:
            actual_rate: Observed deletion rate this step (fraction in [0, 1]).
            return_state: If True, also return a diagnostic dict.

        Returns:
            alpha (float) if return_state is False.
            (alpha, state_dict) if return_state is True, where state_dict contains:
                - "error": target_rate - actual_rate
                - "p_acc": current proportional accumulator
                - "i_acc": current integral accumulator
                - "alpha": the returned alpha value
                - "actual_rate": the actual_rate passed in
        """
        error = self.target_rate - actual_rate

        self.p_acc = self.gamma * self.p_acc + (1 - self.gamma) * self.kp * error
        self.i_acc = self.i_acc + self.ki * error

        alpha = max(0.0, self.p_acc + self.i_acc)

        if return_state:
            return alpha, {
                "error": error,
                "p_acc": self.p_acc,
                "i_acc": self.i_acc,
                "alpha": alpha,
                "actual_rate": actual_rate,
            }
        return alpha

    def reset(self):
        """Reset accumulator state (useful between train/eval phases)."""
        self.p_acc = 0.0
        self.i_acc = 0.0