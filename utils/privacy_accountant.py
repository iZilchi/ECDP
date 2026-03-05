import math

class RDPAccountant:
    """Simple RDP accountant for Gaussian mechanism."""
    def __init__(self, delta):
        self.delta = delta
        self.rdp_orders = list(range(2, 64))  # orders for composition
        self.rdp_buffer = {alpha: 0.0 for alpha in self.rdp_orders}

    def compute_rdp(self, sigma, steps, orders):
        """RDP for Gaussian mechanism with sampling rate q=1.0 (full batch)."""
        # For full batch (no subsampling), RDP = steps * alpha / (2 sigma^2)
        return [steps * alpha / (2 * sigma**2) for alpha in orders]

    def add_round(self, sigma):
        """Add one round with given noise multiplier."""
        rdp_values = self.compute_rdp(sigma, 1, self.rdp_orders)
        for i, alpha in enumerate(self.rdp_orders):
            self.rdp_buffer[alpha] += rdp_values[i]

    def get_epsilon(self):
        """Convert RDP to (ε,δ)-DP."""
        eps = []
        for alpha in self.rdp_orders:
            eps.append(self.rdp_buffer[alpha] + math.log(1/self.delta) / (alpha - 1))
        return min(eps)

    def reset(self):
        self.rdp_buffer = {alpha: 0.0 for alpha in self.rdp_orders}