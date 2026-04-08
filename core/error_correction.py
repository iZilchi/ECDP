import torch
import numpy as np


class ErrorCorrection:
    """
    Post-aggregation error correction for DP-FL.

    Improvements over the original
    --------------------------------
    1. **Layer-wise statistics** — mean/std are tracked per tensor *shape*
       bucket (conv vs fc layers behave very differently under DP noise).
       Computing a global mean/std across all parameters mixes large conv
       weight tensors with small bias vectors, making EVC clip around the
       wrong centre.

    2. **Adaptive momentum** — trusts fresh data more in early rounds
       (effective_momentum = min(base, 1 - 1/round)).

    3. **reset()** — wipes stale state; called by ECDPFL when collapse is
       detected.

    4. **Gradient-sign consistency check** — if the sign of the corrected
       gradient is globally opposed to the raw gradient, the correction is
       doing more harm than good and we fall back to just AGS without EVC.
    """

    def __init__(self, momentum: float = 0.9):
        self.momentum      = momentum
        self.mean_running  = {}   # key -> exponential moving average of mean
        self.std_running   = {}   # key -> EMA of std
        self._round_count  = {}   # key -> number of rounds seen

    def reset(self):
        """Wipe all running statistics."""
        self.mean_running.clear()
        self.std_running.clear()
        self._round_count.clear()

    def _update_stats(self, key: str, tensor: torch.Tensor):
        new_mean = tensor.mean().item()
        new_std  = float(tensor.std().item())

        if key not in self.mean_running:
            self.mean_running[key] = new_mean
            self.std_running[key]  = max(new_std, 1e-8)
            self._round_count[key] = 1
        else:
            n = self._round_count[key] + 1
            self._round_count[key] = n
            # Adaptive: trust fresh observations more in early rounds
            m = min(self.momentum, 1.0 - 1.0 / n)
            self.mean_running[key] = m * self.mean_running[key] + (1 - m) * new_mean
            self.std_running[key]  = max(
                m * self.std_running[key] + (1 - m) * new_std, 1e-8
            )

    def apply(
        self,
        noisy_update: dict,
        alpha: float,
        c: float,
        use_evc: bool = True,
        use_ags: bool = True,
    ) -> dict:
        corrected = {}

        for key, tensor in noisy_update.items():
            self._update_stats(key, tensor)

            mu  = self.mean_running[key]
            sig = self.std_running[key]

            # ---- Extreme Value Clipping (EVC) ----
            if use_evc:
                lower   = mu - c * sig
                upper   = mu + c * sig
                clipped = torch.clamp(tensor, lower, upper)

                # Sign-consistency guard: if clipping flipped the overall
                # direction of this parameter tensor, the running stats are
                # still poisoned from a bad round -- skip EVC this key.
                orig_sign    = tensor.mean().item()
                clipped_sign = clipped.mean().item()
                if orig_sign != 0 and (orig_sign * clipped_sign < 0):
                    clipped = tensor   # fall back: no EVC this round
            else:
                clipped = tensor

            # ---- Adaptive Gradient Smoothing (AGS) ----
            if use_ags:
                corrected[key] = alpha * clipped + (1.0 - alpha) * tensor
            else:
                corrected[key] = clipped

        return corrected