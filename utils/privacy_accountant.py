import math

class RDPAccountant:
    def __init__(self, delta):
        self.delta = delta
        self.rdp_orders = list(range(2, 64))
        self.rdp_buffer = {alpha: 0.0 for alpha in self.rdp_orders}

    def compute_rdp(self, sigma, steps, orders):
        return [steps * alpha / (2 * sigma**2) for alpha in orders]

    def add_round(self, sigma):
        rdp_values = self.compute_rdp(sigma, 1, self.rdp_orders)
        for i, alpha in enumerate(self.rdp_orders):
            self.rdp_buffer[alpha] += rdp_values[i]

    def get_epsilon(self):
        eps = []
        for alpha in self.rdp_orders:
            eps.append(self.rdp_buffer[alpha] + math.log(1/self.delta) / (alpha - 1))
        return min(eps)

    def reset(self):
        self.rdp_buffer = {alpha: 0.0 for alpha in self.rdp_orders}