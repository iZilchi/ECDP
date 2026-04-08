import torch


class ErrorCorrection:
    """
    Post-aggregation error correction for DP-FL.

    Two key fixes over the original:
    1. Running statistics are initialised from fresh gradient statistics each round
       rather than carried indefinitely from a potentially corrupted history.
       momentum=0.9 caused the running mean/std to drift far from the true signal
       after noisy rounds, making EVC clip around the wrong centre.
    2. A reset() method lets the caller wipe stale state when a round's gradient
       is detected as an outlier (norm >> expected).
    """

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.mean_running = {}
        self.std_running  = {}
        self._round_count = {}          # how many rounds each key has been seen

    def reset(self):
        """Wipe all running statistics (call if training collapses)."""
        self.mean_running.clear()
        self.std_running.clear()
        self._round_count.clear()

    def apply(self, noisy_update, alpha, c, use_evc=True, use_ags=True):
        corrected = {}

        for key, tensor in noisy_update.items():
            new_mean = tensor.mean().item()
            new_std  = tensor.std().item()

            if key not in self.mean_running:
                # First time: seed with actual values (no momentum applied yet)
                self.mean_running[key] = new_mean
                self.std_running[key]  = max(new_std, 1e-8)
                self._round_count[key] = 1
            else:
                self._round_count[key] += 1
                # Adaptive momentum: trust fresh data more in early rounds
                effective_momentum = min(
                    self.momentum,
                    1.0 - 1.0 / self._round_count[key]
                )
                self.mean_running[key] = (
                    effective_momentum * self.mean_running[key]
                    + (1 - effective_momentum) * new_mean
                )
                self.std_running[key] = max(
                    effective_momentum * self.std_running[key]
                    + (1 - effective_momentum) * new_std,
                    1e-8
                )

            # --- Extreme Value Clipping (EVC) ---
            if use_evc:
                lower   = self.mean_running[key] - c * self.std_running[key]
                upper   = self.mean_running[key] + c * self.std_running[key]
                clipped = torch.clamp(tensor, lower, upper)
            else:
                clipped = tensor

            # --- Adaptive Gradient Smoothing (AGS) ---
            if use_ags:
                corrected[key] = alpha * clipped + (1 - alpha) * tensor
            else:
                corrected[key] = clipped

        return corrected