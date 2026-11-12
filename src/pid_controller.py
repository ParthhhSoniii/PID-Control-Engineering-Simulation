"""
PID controller for the mass-spring-damper simulation.

The controller calculates a force from position error.
It does not update the plant; simulation.py passes the returned
force to plant.external_force.
"""


class PIDController:
    def __init__(self, kp, ki, kd, setpoint,output_limits=None,anti_windup=False):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)

        # Optional actuator limits: (minimum_force, maximum_force)
        self.output_limits = output_limits
        self.anti_windup = anti_windup

        # Internal memory for integral and derivative calculations.
        self.previous_error = None
        self.integral = 0.0
        self.derivative = 0.0    

        # Latest values, useful for debugging and plotting.
        self.error = 0.0
        self.proportional_term = 0.0 # P
        self.integral_term = 0.0 # I
        self.derivative_term = 0.0  # D
        self.output = 0.0

    def calculate_error(self, current_position):
        """Calculate target position minus current position."""
        self.error = self.setpoint - float(current_position)
        return self.error

    def calculate_output(self, current_position, dt):
        if dt <= 0:
            raise ValueError("dt must be greater than zero.")

        # 1. Calculate error
        self.error = self.setpoint - current_position

        # 2. Proportional term
        self.proportional = self.kp * self.error
        
        # 3. Derivative term
        if self.previous_error is None:
            # First iteration: no previous error available
            self.derivative = 0.0
        else:
            self.derivative = (
                self.error - self.previous_error
            ) / dt

        self.derivative_term = self.kd * self.derivative

        # 4. Calculate candidate integral
        candidate_integral = self.integral + self.error * dt
        candidate_integral_term = self.ki * candidate_integral

        # 5. Calculate controller output before saturation
        candidate_raw_output = (
            self.proportional
            + candidate_integral_term
            + self.derivative_term
        )

        # 6. Conditional integration (anti-windup)
        if self.anti_windup and self.output_limits is not None:

            minimum_force, maximum_force = self.output_limits

            pushing_further_into_saturation = (  # Condition for anti-windup
                (candidate_raw_output > maximum_force and self.error > 0)
                or
                (candidate_raw_output < minimum_force and self.error < 0)
            )

            if not pushing_further_into_saturation: # Allow integration when it is safe
                self.integral = candidate_integral

        else:
            # Normal integration: no anti-windup
            self.integral = candidate_integral

        # 7. Recalculate output using the accepted integral
        self.integral_term = self.ki * self.integral   # when anti-windup is enabled self.integral  = previous_integral if the output 

        self.raw_output = (
            self.proportional
            + self.integral_term
            + self.derivative_term
        )

        # 8. Apply actuator limits
        if self.output_limits is not None:
            minimum_force, maximum_force = self.output_limits

            self.output = max(
                minimum_force,
                min(self.raw_output, maximum_force)
            )
        else:
            self.output = self.raw_output

        self.previous_error = self.error

        return self.output

    def reset(self):
        """Clear controller history before starting a new experiment."""
        self.error = 0.0
        self.previous_error = None

        self.integral = 0.0

        self.proportional = 0.0
        self.integral_term = 0.0
        self.derivative = 0.0
        self.derivative_term = 0.0

        self.raw_output = 0.0
        self.output = 0.0


if __name__ == "__main__":
    # With Ki=0 and Kd=0, this is a simple P-controller check.
    controller = PIDController(
        kp=200.0,
        ki=0.0,
        kd=0.0,
        setpoint=1.0,
    )

    force = controller.calculate_output(
        current_position=0.0,
        dt=0.001,
    )

    print(f"Position error: {controller.error:.3f} m")
    print(f"Control force: {force:.3f} N")
