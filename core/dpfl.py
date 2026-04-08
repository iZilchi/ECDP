import torch
import numpy as np
from .federated_learning import FederatedLearningBase
from .differential_privacy import DifferentialPrivacy
from .error_correction import ErrorCorrection
from utils.privacy_accountant import RDPAccountant


class BasicDPFL(FederatedLearningBase):
    def __init__(self, num_clients, model_class, device,
                 epsilon=None, delta=1e-5, clip_norm=1.0,
                 target_epsilon=None, max_rounds=100):
        super().__init__(num_clients, model_class, device)
        assert (epsilon is not None) or (target_epsilon is not None), \
            "Either epsilon or target_epsilon must be provided"
        self.epsilon        = epsilon
        self.epsilon_target = target_epsilon
        self.delta          = delta
        self.clip_norm      = clip_norm
        self.num_clients    = num_clients
        self.max_rounds     = max_rounds
        self.accountant     = RDPAccountant(delta)
        self.noise_scale    = None
        self.dp             = DifferentialPrivacy(epsilon=1.0, delta=delta,
                                                   clip_norm=clip_norm)

    def _compute_noise_scale(self):
        """
        Noise is added to the *average* of client updates (sensitivity = C/K),
        so σ must account for the fact that aggregation divides by num_clients.

        FIX vs original: sensitivity = clip_norm / num_clients  (correct)
        The original code already did this for epsilon_target but applied
        the same formula for fixed epsilon too — kept consistent here.
        """
        sensitivity = self.clip_norm / self.num_clients
        if self.epsilon_target is not None:
            eps_per_round = self.epsilon_target / self.max_rounds
        else:
            eps_per_round = self.epsilon

        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / eps_per_round
        return sigma

    def _train_client_get_update(self, global_weights, dataloader, epochs):
        raw_update = super()._train_client_get_update(
            global_weights, dataloader, epochs)
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
    def __init__(self, num_clients, model_class, device,
                 epsilon=None, delta=1e-5, clip_norm=1.0,
                 target_epsilon=None, max_rounds=100,
                 use_evc=True, use_ags=True,
                 c=2.5, alpha=0.8, correction_momentum=0.9):
        super().__init__(num_clients, model_class, device,
                         epsilon, delta, clip_norm,
                         target_epsilon, max_rounds)
        self.use_evc = use_evc
        self.use_ags = use_ags
        self.c       = c
        self.alpha   = alpha
        self.error_correction = ErrorCorrection(momentum=correction_momentum)

        # Track how noisy each round is so we can detect collapse
        self._prev_global_norm = None

    def _aggregate_updates(self, client_updates):
        # 1. Get the DP-noised average from the parent
        noisy_avg = super()._aggregate_updates(client_updates)

        # 2. Collapse detection: if the aggregated update norm is
        #    drastically larger than the previous round, the noise
        #    has swamped the signal — reset error correction stats
        #    so EVC is centred on the fresh (noisy) gradient rather
        #    than a stale "good" estimate.
        current_norm = sum(
            torch.sum(v ** 2).item() for v in noisy_avg.values()
        ) ** 0.5

        if self._prev_global_norm is not None:
            ratio = current_norm / (self._prev_global_norm + 1e-8)
            if ratio > 10.0:          # norm jumped >10× — likely noise spike
                self.error_correction.reset()

        self._prev_global_norm = current_norm

        # 3. Apply post-aggregation error correction
        corrected_avg = self.error_correction.apply(
            noisy_avg,
            alpha=self.alpha,
            c=self.c,
            use_evc=self.use_evc,
            use_ags=self.use_ags,
        )
        return corrected_avg