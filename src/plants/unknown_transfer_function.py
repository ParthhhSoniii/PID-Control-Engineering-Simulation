import numpy as np


class UnknownTransferFunction:
    '''Black-box plant: input is control force; output is position.

    Internal dynamics are deliberately encapsulated in this class. The
    experiment should only use the plant through the shared simulation API.
    '''

    def __init__(self, initial_position=0.0, initial_velocity=0.0,
                 initial_acceleration=0.0):
        
        self.initial_position = float(initial_position)
        self.initial_velocity = float(initial_velocity)
        self.initial_acceleration = float(initial_acceleration)
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        self.acceleration = self.initial_acceleration
        self.control_input = 0.0

    @property
    def external_force(self):
        # Compatibility with simulation.py implementations that assign
        # plant.external_force = controller_output.
        return self.control_input

    @external_force.setter
    def external_force(self, value):
        self.control_input = float(value)

    def set_input(self, control_input):
        self.control_input = float(control_input)

    def calculate_net_acceleration(self):
        # For this third-order plant, this method returns y'''.
        return (
            self.control_input
            - 3.0 * self.acceleration
            - 3.0 * self.velocity
            - self.position
        )

    def _state_derivative(self, state, control_input):
        y, y_dot, y_ddot = state
        y_ddd = control_input - 3.0 * y_ddot - 3.0 * y_dot - y
        return np.array([y_dot, y_ddot, y_ddd], dtype=float)

    def update_state(self, dt):
        dt = float(dt)
        state = np.array(
            [self.position, self.velocity, self.acceleration], dtype=float
        )
        u = self.control_input

        k1 = self._state_derivative(state, u)
        k2 = self._state_derivative(state + 0.5 * dt * k1, u)
        k3 = self._state_derivative(state + 0.5 * dt * k2, u)
        k4 = self._state_derivative(state + dt * k3, u)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.position = float(next_state[0])
        self.velocity = float(next_state[1])
        self.acceleration = float(next_state[2])

    def reset(self, position=None, velocity=None, acceleration=None):
        self.position = self.initial_position if position is None else float(position)
        self.velocity = self.initial_velocity if velocity is None else float(velocity)
        self.acceleration = (
            self.initial_acceleration if acceleration is None
            else float(acceleration)
        )
        self.control_input = 0.0

