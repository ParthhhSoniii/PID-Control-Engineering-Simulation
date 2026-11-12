"""Mass-spring-damper plant using explicit Euler integration."""


class MassSpringDamper:
    """
    One-dimensional mass-spring-damper system.

    Sign convention:
        Positive = right
        Negative = left
    """

    def __init__(
        self,
        mass,
        damping,
        spring_constant,
        initial_position=0.0,
        initial_velocity=0.0,
        external_force=0.0,
    ):

        if mass <= 0:
            raise ValueError("Mass must be greater than zero.")

        if damping < 0 or spring_constant < 0:
            raise ValueError("Damping and spring constant cannot be negative.")

        self.mass = float(mass)
        self.damping = float(damping)
        self.spring_constant = float(spring_constant)

        self.position = float(initial_position)
        self.velocity = float(initial_velocity)
        self.external_force = float(external_force)

        self.net_acceleration = self.calculate_net_acceleration() 

    def calculate_forces(self):
        """Calculate all forces acting on the mass."""

        spring_force = -self.spring_constant * self.position
        damping_force = -self.damping * self.velocity

        net_force = (
            self.external_force
            + spring_force
            + damping_force
        )

        return {
            "external_force": self.external_force,
            "spring_force": spring_force,
            "damping_force": damping_force,
            "net_force": net_force,
        }

    def calculate_net_acceleration(self):
        """Calculate acceleration using Newton's second law."""

        forces = self.calculate_forces()

        return forces["net_force"] / self.mass

    def update_state(self, dt):
        """Update position and velocity using Euler integration."""

        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")

        current_acceleration = self.calculate_net_acceleration()
        current_forces = self.calculate_forces()

        # Explicit Euler integration
        new_position = self.position + self.velocity * dt
        new_velocity = self.velocity + current_acceleration * dt

        self.position = new_position
        self.velocity = new_velocity

        self.net_acceleration = self.calculate_net_acceleration()

        return {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.net_acceleration,
            **current_forces,
        }

    def simulate(self, duration, dt):
        """Simulate the plant with a constant external force."""

        if duration < 0 or dt <= 0:
            raise ValueError("Invalid duration or time step.")

        number_of_steps = int(duration / dt)

        time = [0.0] # Convert into a list to store time values for creting a time series graph 
        position = [self.position]
        velocity = [self.velocity]
        net_acceleration = [self.net_acceleration]

        initial_forces = self.calculate_forces()

        external_force = [initial_forces["external_force"]]
        spring_force = [initial_forces["spring_force"]]
        damping_force = [initial_forces["damping_force"]]
        net_force = [initial_forces["net_force"]]

        for step in range(number_of_steps):

            result = self.update_state(dt)

            time.append((step + 1) * dt) #initially step is 0
            position.append(result["position"])
            velocity.append(result["velocity"])
            net_acceleration.append(result["acceleration"])

            external_force.append(result["external_force"])
            spring_force.append(result["spring_force"])
            damping_force.append(result["damping_force"])
            net_force.append(result["net_force"])

        return {
            "time": time,
            "position": position,
            "velocity": velocity,
            "acceleration": net_acceleration,
            "external_force": external_force,
            "spring_force": spring_force,
            "damping_force": damping_force,
            "net_force": net_force,
        }

    def reset(
        self,
        position=0.0,
        velocity=0.0,
        external_force=0.0,
    ):
        """Reset the plant for another experiment."""

        self.position = float(position)
        self.velocity = float(velocity)
        self.external_force = float(external_force)

        self.net_acceleration = self.calculate_net_acceleration()


if __name__ == "__main__":

    plant = MassSpringDamper(
        mass=1.0,
        damping=10.0,
        spring_constant=20.0,
        initial_position=0.0,
        initial_velocity=0.0,
        external_force=5.0, # right now manually given just to test in pipeline it's completely handled by the  PID controller (feedback force)
    )

    results = plant.simulate(
        duration=20.0,
        dt=0.001
    )

    print("Final Position:", results["position"][-1])
    print("Final Velocity:", results["velocity"][-1])
    print("Final Acceleration:", results["acceleration"][-1])