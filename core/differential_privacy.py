import torch
import numpy as np


class DifferentialPrivacy:
    """
    Gaussian-mechanism DP for federated learning.

    Key design note
    ---------------
    Noise is added to the *averaged* aggregate (sensitivity = C / K where K =
    num_clients).  This class operates on a single update dict and does NOT
    divide by K — that division happens implicitly when FederatedLearningBase
    averages the client updates before passing the result to add_noise().

    The caller (BasicDPFL._compute_noise_scale) is responsible for setting
    sigma = (C / K) * sqrt(2 ln(1.25/δ)) / ε.
    """

    def __init__(self, epsilon, delta=1e-5, clip_norm=1.0):
        self.epsilon   = epsilon
        self.delta     = delta
        self.clip_norm = clip_norm

    # ------------------------------------------------------------------
    # noise sigma for a *single* update (before averaging) — kept for
    # backward-compat but not used by BasicDPFL
    # ------------------------------------------------------------------
    def calculate_noise_scale(self):
        return (
            self.clip_norm
            * np.sqrt(2 * np.log(1.25 / self.delta))
            / self.epsilon
        )

    # ------------------------------------------------------------------
    # Gradient clipping  (L2 norm bound)
    # ------------------------------------------------------------------
    def clip_update(self, update):
        squared_sum = sum(torch.sum(p ** 2).item() for p in update.values())
        total_norm  = np.sqrt(squared_sum)

        if total_norm > self.clip_norm:
            scale = self.clip_norm / (total_norm + 1e-6)
            update = {k: v * scale for k, v in update.items()}
        return update

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------
    def add_noise(self, update, noise_scale):
        """
        Add independent Gaussian noise N(0, noise_scale²) to every parameter.

        noise_scale should already encode sensitivity / epsilon so that the
        resulting mechanism satisfies (ε, δ)-DP after averaging K clipped
        updates.
        """
        noisy_update = {}
        for k, v in update.items():
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=v.shape,
                device=v.device,
            )
            noisy_update[k] = v + noise
        return noisy_update