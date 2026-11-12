import numpy as np


class PositionSensor:

    def __init__(self, noise_limit=0.1, seed=42): # noise_limit --> 0.1 meters = 10 cm

        self.noise_limit = noise_limit
        self.rng = np.random.default_rng(seed)

    def read(self, actual_position):

        noise = self.rng.uniform(
            -self.noise_limit,
            self.noise_limit
        )

        measured_position = actual_position + noise

        return measured_position