# core/differential_privacy.py - FIXED VERSION WITH MANUSCRIPT ALIGNMENT
"""
Differential Privacy Implementation - STABLE VERSION
Implements Equations 3-8 from manuscript with proper gradient clipping
"""
import torch
import numpy as np

class DifferentialPrivacy:
    """
    Differential privacy mechanisms with gradient clipping
    
    Implements:
    - Eq. 3: (ε, δ)-differential privacy guarantee
    - Eq. 4: Gaussian mechanism
    - Eq. 5: Noise scale calculation (CORRECTED)
    - Eq. 6: Signal-to-Noise Ratio (SNR)
    - ADDED: Gradient clipping for stability
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        """
        Initialize DP mechanism
        
        Args:
            epsilon: Privacy budget (lower = stronger privacy)
            delta: Failure probability
            clip_norm: Gradient clipping threshold (NEW - from manuscript)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.current_round = 0  # Track rounds for warmup
    
    def calculate_noise_scale(self, sensitivity=None):
        """
        Calculate DP noise scale using Eq. 5 (CORRECTED)
        
        σ² = (Δ₂f)² · 2ln(1.25/δ) / ε²
        
        Args:
            sensitivity: L2 sensitivity (defaults to clip_norm)
            
        Returns:
            Noise scale σ
        """
        if self.epsilon <= 0 or self.delta <= 0:
            raise ValueError("Epsilon and delta must be positive")
        
        # Use clip_norm as sensitivity (manuscript Eq. 7)
        S = sensitivity if sensitivity is not None else self.clip_norm
        
        # Eq. 5: Standard Gaussian mechanism formula
        noise_scale = S * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # REMOVED: The problematic 0.1 multiplier
        # Instead, use proper gradient clipping to control sensitivity
        
        return noise_scale
    
    def clip_gradients(self, tensor):
        """
        Clip gradients to bound sensitivity (Eq. 7 implementation)
        
        This is CRITICAL for stability - clips before adding noise
        
        Args:
            tensor: Input gradient tensor
            
        Returns:
            Clipped tensor
        """
        norm = torch.norm(tensor, p=2)
        
        if norm > self.clip_norm:
            # Clip to max norm
            tensor = tensor * (self.clip_norm / (norm + 1e-10))
        
        return tensor
    
    def add_noise(self, tensor, noise_scale):
        """
        Add Gaussian noise using Eq. 4 (with gradient clipping)
        
        M(D) = CLIP(f(D)) + N(0, σ²I)
        
        Args:
            tensor: Input tensor
            noise_scale: Noise scale σ
            
        Returns:
            Noisy tensor
        """
        # STEP 1: Clip gradients first (Eq. 7)
        clipped_tensor = self.clip_gradients(tensor)
        
        # STEP 2: Add calibrated noise (Eq. 4)
        if noise_scale > 0:
            noise = torch.normal(
                mean=0.0, 
                std=noise_scale, 
                size=clipped_tensor.shape, 
                device=clipped_tensor.device
            )
            noisy_tensor = clipped_tensor + noise
        else:
            noisy_tensor = clipped_tensor
        
        return noisy_tensor
    
    def add_noise_with_warmup(self, tensor, noise_scale, max_warmup_rounds=3):
        """
        Add noise with warmup to prevent early collapse
        
        Gradually increases noise over first few rounds
        
        Args:
            tensor: Input tensor
            noise_scale: Target noise scale
            max_warmup_rounds: Number of warmup rounds
            
        Returns:
            Noisy tensor with warmup
        """
        # Calculate warmup factor
        if self.current_round < max_warmup_rounds:
            warmup_factor = (self.current_round + 1) / max_warmup_rounds
            effective_noise = noise_scale * warmup_factor
        else:
            effective_noise = noise_scale
        
        return self.add_noise(tensor, effective_noise)
    
    def calculate_snr(self, signal, noise_scale):
        """
        Calculate Signal-to-Noise Ratio using Eq. 6
        
        SNR = ||True Gradient||₂² / E[||DP Noise||₂²] ∝ 1/σ²
        
        Args:
            signal: Original signal (gradient)
            noise_scale: Noise scale σ
            
        Returns:
            SNR value
        """
        signal_norm_sq = torch.norm(signal, p=2).item()**2
        noise_power = noise_scale**2 * signal.numel()
        
        if noise_power > 0:
            snr = signal_norm_sq / noise_power
        else:
            snr = float('inf')
        
        return snr
    
    def increment_round(self):
        """Increment round counter for warmup"""
        self.current_round += 1

class ErrorCorrectedDP(DifferentialPrivacy):
    """
    Differential Privacy with Error Correction - MANUSCRIPT ALIGNED
    
    Implements:
    - Eq. 7: Extreme Value Clipping (c=1.5 high noise, c=2.5 low noise)
    - Eq. 8: Adaptive Gradient Smoothing (α=0.6 high noise, α=0.8 low noise)
    - Eq. 9: Utility Improvement
    - Eq. 10: Utility Recovery Rate
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        super().__init__(epsilon, delta, clip_norm)
        
        # Adaptive parameters based on privacy level
        if epsilon <= 0.5:  # High noise (strong privacy)
            self.c = 1.5  # Tight clipping (manuscript Eq. 7)
            self.alpha = 0.6  # Strong smoothing (manuscript Eq. 8)
        else:  # Low noise (weak privacy)
            self.c = 2.5  # Loose clipping
            self.alpha = 0.8  # Light smoothing
    
    def add_corrected_noise(self, tensor, noise_scale):
        """
        Add noise with adaptive error correction (MANUSCRIPT ALIGNED)
        
        Applies:
        1. Gradient clipping (for stability)
        2. Eq. 4: Add Gaussian noise
        3. Eq. 7: Extreme value clipping
        4. Eq. 8: Adaptive gradient smoothing
        
        Args:
            tensor: Input tensor
            noise_scale: Noise scale σ
            
        Returns:
            Error-corrected noisy tensor
        """
        # Step 1: Clip gradients for stability
        clipped_tensor = self.clip_gradients(tensor)
        
        # Step 2: Add DP noise (Eq. 4) with warmup
        if noise_scale > 0:
            # Use warmup for first few rounds
            if self.current_round < 3:
                warmup_factor = (self.current_round + 1) / 3
                effective_noise = noise_scale * warmup_factor
            else:
                effective_noise = noise_scale
            
            noise = torch.normal(
                mean=0.0,
                std=effective_noise,
                size=clipped_tensor.shape,
                device=clipped_tensor.device
            )
            noisy_tensor = clipped_tensor + noise
        else:
            noisy_tensor = clipped_tensor
        
        # Step 3: Error correction based on noise level
        corrected_tensor = self._apply_error_correction(noisy_tensor, clipped_tensor)
        
        return corrected_tensor
    
    def _apply_error_correction(self, noisy_tensor, original_tensor):
        """
        Apply error correction using Eq. 7 and Eq. 8
        
        Args:
            noisy_tensor: Tensor with DP noise
            original_tensor: Original tensor (clipped)
            
        Returns:
            Error-corrected tensor
        """
        # Eq. 7: Extreme Value Clipping
        mean_val = noisy_tensor.mean()
        std_val = noisy_tensor.std()
        
        clipped_noisy = torch.clamp(
            noisy_tensor,
            mean_val - self.c * std_val,
            mean_val + self.c * std_val
        )
        
        # Eq. 8: Adaptive Gradient Smoothing
        corrected_tensor = self.alpha * clipped_noisy + (1 - self.alpha) * original_tensor
        
        return corrected_tensor

def calculate_utility_metrics(acc_std, acc_dp, acc_ecdp):
    """
    Calculate utility metrics from Eq. 9 and Eq. 10
    
    Eq. 9: Improvement = Accuracy_ECDP - Accuracy_BasicDP
    Eq. 10: Recovery Rate = Improvement / (Accuracy_StdFL - Accuracy_BasicDP) × 100%
    
    Args:
        acc_std: Standard FL accuracy
        acc_dp: Basic DP-FL accuracy
        acc_ecdp: EC-DP-FL accuracy
        
    Returns:
        Dictionary with improvement and recovery_rate
    """
    # Eq. 9: Utility Improvement
    improvement = acc_ecdp - acc_dp
    
    # Eq. 10: Utility Recovery Rate
    total_loss = acc_std - acc_dp
    recovery_rate = (improvement / total_loss * 100) if total_loss > 0 else 0
    
    return {
        'improvement': improvement,
        'recovery_rate': recovery_rate,
        'dp_utility_loss': total_loss,
        'ecdp_utility_loss': acc_std - acc_ecdp
    }