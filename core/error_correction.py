# core/error_correction.py
import torch

class ErrorCorrection:
    """Error correction algorithms for DP noise"""
    
    @staticmethod
    def gradient_smoothing(gradients, window_size=3):
        """Apply smoothing to gradients"""
        if len(gradients) < window_size:
            return gradients[-1] if gradients else None
        
        # Simple moving average
        smoothed = torch.stack(gradients[-window_size:]).mean(dim=0)
        return smoothed
    
    @staticmethod
    def extreme_value_clipping(tensor, threshold=2.0):
        """Clip extreme values"""
        mean = tensor.mean()
        std = tensor.std()
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        return torch.clamp(tensor, lower_bound, upper_bound)