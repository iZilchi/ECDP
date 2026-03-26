import torch
import numpy as np

class DifferentialPrivacy:
    def __init__(self, epsilon, delta=1e-5, clip_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

    def calculate_noise_scale(self):
        """Eq. 5: σ = C · √(2·ln(1.25/δ)) / ε"""
        return self.clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_update(self, update):
        squared_sum = 0.0
        # Collect keys of float tensors
        float_keys = []
        for k, p in update.items():
            if p.dtype in (torch.float, torch.float32, torch.float64):
                squared_sum += torch.sum(p ** 2).item()
                float_keys.append(k)
        
        total_norm = np.sqrt(squared_sum)
        
        if total_norm > self.clip_norm:
            scale = self.clip_norm / (total_norm + 1e-6)
            for k in float_keys:
                update[k] = update[k] * scale
        return update

    def add_noise(self, update, noise_scale):
        noisy_update = {}
        for k, v in update.items():
            noise = torch.normal(mean=0.0, std=noise_scale, size=v.shape, device=v.device)
            noisy_update[k] = v + noise
        return noisy_update