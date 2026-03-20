import torch
import numpy as np
from .federated_learning import FederatedLearningBase
from .differential_privacy import DifferentialPrivacy
from .error_correction import ErrorCorrection
from utils.privacy_accountant import RDPAccountant

class BasicDPFL(FederatedLearningBase):
    def __init__(self, num_clients, model_class, device, participation_rate=1.0,
                 epsilon=None, delta=1e-5, clip_norm=1.0,
                 target_epsilon=None, max_rounds=100):
        super().__init__(num_clients, model_class, device, participation_rate)
        assert (epsilon is not None) or (target_epsilon is not None), \
            "Either epsilon or target_epsilon must be provided"
        self.epsilon = epsilon
        self.epsilon_target = target_epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.max_rounds = max_rounds
        self.accountant = RDPAccountant(delta)
        self.noise_scale = None
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=delta, clip_norm=clip_norm)

    def _compute_noise_scale(self):
        sensitivity = self.clip_norm / self.num_clients
        if self.epsilon_target is not None:
            eps_per_round = self.epsilon_target / self.max_rounds
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / eps_per_round
            return sigma
        else:
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            return sigma

    def _train_client_get_update(self, global_weights, dataloader, epochs):
        raw_update = super()._train_client_get_update(global_weights, dataloader, epochs)
        clipped = self.dp.clip_update(raw_update)
        return clipped

    def _aggregate_updates(self, client_updates):
        avg_update = super()._aggregate_updates(client_updates)
        if self.noise_scale is None:
            self.noise_scale = self._compute_noise_scale()
        self.accountant.add_round(self.noise_scale)
        noisy_avg = self.dp.add_noise(avg_update, self.noise_scale)
        return noisy_avg

    def get_spent_epsilon(self):
        return self.accountant.get_epsilon()


class ECDPFL(BasicDPFL):
    def __init__(self, num_clients, model_class, device, participation_rate=1.0,
                 epsilon=None, delta=1e-5, clip_norm=1.0,
                 target_epsilon=None, max_rounds=100,
                 c=2.5, alpha=0.8, warm_up=0, correction_momentum=0.9,
                 use_clipping=True, use_smoothing=True):
        super().__init__(num_clients, model_class, device, participation_rate,
                         epsilon, delta, clip_norm,
                         target_epsilon, max_rounds)
        self.c = c
        self.alpha = alpha
        self.warm_up = warm_up
        self.use_clipping = use_clipping
        self.use_smoothing = use_smoothing
        self.error_correction = ErrorCorrection(momentum=correction_momentum)

    def _aggregate_updates(self, client_updates):
        # Standard aggregation (FedAvg) with DP
        aggregated = super()._aggregate_updates(client_updates)   # aggregates client updates (no noise yet)
        if self.noise_scale is None:
            self.noise_scale = self._compute_noise_scale()
        self.accountant.add_round(self.noise_scale)
        noisy_avg = self.dp.add_noise(aggregated, self.noise_scale)

        # Apply post‑aggregation error correction
        corrected_avg = self.error_correction.apply(
            noisy_avg,
            alpha=self.alpha,
            c=self.c,
            warm_up_rounds=self.warm_up,
            use_clipping=self.use_clipping,
            use_smoothing=self.use_smoothing
        )
        return corrected_avg