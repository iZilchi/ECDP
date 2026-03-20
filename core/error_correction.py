import torch

class ErrorCorrection:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.mean_running = {}
        self.std_running = {}
        self.counter = 0

    def apply(self, noisy_update, alpha, c, warm_up_rounds=0,
              use_clipping=True, use_smoothing=True):
        corrected = {}
        for key, tensor in noisy_update.items():
            if key not in self.mean_running:
                self.mean_running[key] = tensor.mean().item()
                self.std_running[key] = tensor.std().item()
            else:
                new_mean = tensor.mean().item()
                new_std = tensor.std().item()
                self.mean_running[key] = (self.momentum * self.mean_running[key] +
                                          (1 - self.momentum) * new_mean)
                self.std_running[key] = (self.momentum * self.std_running[key] +
                                         (1 - self.momentum) * new_std)

            if use_clipping:
                lower = self.mean_running[key] - c * self.std_running[key]
                upper = self.mean_running[key] + c * self.std_running[key]
                clipped = torch.clamp(tensor, lower, upper)
            else:
                clipped = tensor

            if use_smoothing:
                if self.counter < warm_up_rounds:
                    corrected[key] = tensor
                else:
                    corrected[key] = alpha * clipped + (1 - alpha) * tensor
            else:
                corrected[key] = clipped

        self.counter += 1
        return corrected