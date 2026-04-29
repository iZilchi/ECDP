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
        self.dp             = DifferentialPrivacy(
            epsilon=1.0, delta=delta, clip_norm=clip_norm)

    def _compute_noise_scale(self):
        """
        sigma = (C / K) * sqrt(2 ln(1.25/delta)) / epsilon_per_round

        C   = clip_norm   (L2 sensitivity of one client update)
        K   = num_clients (averaging divides by K, reducing sensitivity)
        """
        sensitivity = self.clip_norm / self.num_clients
        if self.epsilon_target is not None:
            eps_per_round = self.epsilon_target / self.max_rounds
        else:
            eps_per_round = self.epsilon
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / eps_per_round

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
    """
    Error-Corrected DP-FL.

    Collapse detection
    ------------------
    Two triggers cause error_correction.reset():

    1. Norm spike  — update L2 norm jumps >10x vs previous round.
       Catches noise spikes that happen before the model has learned anything.

    2. Direction flip — the best-class prediction (argmax of mean logit) of
       the *corrected* update differs from the noisy update by more than
       `_flip_threshold` fraction of parameters.  This catches the case where
       EVC clips toward a stale mean and actively steers the model wrong.
       Implemented cheaply: compare sign-of-mean for each layer key.
    """

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

        self._prev_global_norm = None
        self._flip_threshold   = 0.5    # if >50% of layers flip sign, reset

    def _aggregate_updates(self, client_updates):
        # 1. DP-noised average from parent
        noisy_avg = super()._aggregate_updates(client_updates)

        # 2. Norm-spike collapse detection
        current_norm = sum(
            torch.sum(v ** 2).item() for v in noisy_avg.values()
        ) ** 0.5

        if self._prev_global_norm is not None:
            ratio = current_norm / (self._prev_global_norm + 1e-8)
            if ratio > 10.0:
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

        # 4. Direction-flip collapse detection (post-correction check)
        flipped = 0
        total   = 0
        for key in noisy_avg:
            orig_sign = noisy_avg[key].mean().item()
            corr_sign = corrected_avg[key].mean().item()
            if orig_sign != 0:
                total += 1
                if orig_sign * corr_sign < 0:
                    flipped += 1

        if total > 0 and (flipped / total) > self._flip_threshold:
            # Correction is steering away from the gradient direction —
            # reset stats and return the uncorrected noisy average this round
            self.error_correction.reset()
            return noisy_avg

        return corrected_avg
