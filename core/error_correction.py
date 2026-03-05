import torch

class ErrorCorrection:
    """
    Error correction with running statistics across rounds.
    Eq. 7 (clipping) and Eq. 8 (smoothing) from the manuscript.
    """
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.mean_running = {}
        self.std_running = {}
        self.counter = 0  # rounds seen

    def apply(self, noisy_update, alpha, c, warm_up_rounds=5):
        """
        Apply correction after warm-up.
        - warm_up_rounds: number of initial rounds to skip correction.
        """
        corrected = {}
        for key, tensor in noisy_update.items():
            # Initialize running stats if first time
            if key not in self.mean_running:
                self.mean_running[key] = tensor.mean().item()
                self.std_running[key] = tensor.std().item()
            else:
                # Update running stats with momentum
                new_mean = tensor.mean().item()
                new_std = tensor.std().item()
                self.mean_running[key] = (self.momentum * self.mean_running[key] +
                                          (1 - self.momentum) * new_mean)
                self.std_running[key] = (self.momentum * self.std_running[key] +
                                         (1 - self.momentum) * new_std)

            # Apply extreme value clipping (Eq. 7)
            lower = self.mean_running[key] - c * self.std_running[key]
            upper = self.mean_running[key] + c * self.std_running[key]
            clipped = torch.clamp(tensor, lower, upper)

            # Apply adaptive smoothing (Eq. 8)
            if self.counter < warm_up_rounds:
                # No correction during warm-up
                corrected[key] = tensor
            else:
                corrected[key] = alpha * clipped + (1 - alpha) * tensor

        self.counter += 1
        return corrected