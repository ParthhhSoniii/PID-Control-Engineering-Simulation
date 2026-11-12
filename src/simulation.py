"""
Closed-loop simulation connecting a MassSpringDamper plant
with a PIDController.

The plant handles physics.
The PID controller calculates the force.
This module coordinates their interaction.


Its responsibilities:
1. Read the plant's current position.
2. Pass the position to the PID controller.
3. Receive the calculated control force.
4. Assign the force to plant.external_force.
5. Update the plant using Euler integration.
6. Record the simulation results.
"""

from src.sensors.position_sensor import PositionSensor

def run_simulation(plant, controller, duration, dt, position_sensor_obj=None):

    if duration < 0:
        raise ValueError("Duration cannot be negative.")

    if dt <= 0:
        raise ValueError("Time step must be positive.")

    controller.reset()

    number_of_steps = int(round(duration / dt))

    time = []
    position = []              # Actual plant position
    measured_position = []     # Sensor reading

    velocity = []
    acceleration = []
    error = []

    # Actual force applied to the plant
    control_force = []

    # Additional controller signals
    raw_control_force = []
    proportional = []
    integral_term = []
    derivative_term = []

    for step in range(number_of_steps + 1):

        current_time = step * dt

        # 1. Read actual plant position
        actual_position = plant.position

        # 2. Obtain sensor measurement
        if position_sensor_obj:
            sensor_reading = position_sensor_obj.read(actual_position)
        else:
            sensor_reading = actual_position

        # 3. Controller uses measured position
        force = controller.calculate_output(
            current_position=sensor_reading,
            dt=dt
        )

        # 4. Apply force to the plant
        plant.external_force = force

        # 5. Record actual and measured states separately
        time.append(current_time)

        position.append(actual_position)
        measured_position.append(sensor_reading)

        velocity.append(plant.velocity)
        acceleration.append(plant.calculate_net_acceleration())

        error.append(controller.error)

        control_force.append(force)

        raw_control_force.append(
            getattr(controller, "raw_output", force)
        )

        proportional.append(
            getattr(controller, "proportional", 0.0)
        )

        integral_term.append(
            getattr(controller, "integral_term", 0.0)
        )

        derivative_term.append(
            getattr(controller, "derivative_term", 0.0)
        )

        # 6. Update plant
        if step < number_of_steps:
            plant.update_state(dt)

    return {
        "time": time,
        "position": position,
        "measured_position": measured_position,
        "velocity": velocity,
        "acceleration": acceleration,
        "error": error,
        "control_force": control_force,
        "raw_control_force": raw_control_force,
        "proportional": proportional,
        "integral_term": integral_term,
        "derivative_term": derivative_term
    }


if __name__ == "__main__":
    # Run from the project root with: python -m src.simulation
    from plants.mass_spring_damper import MassSpringDamper
    from pid_controller import PIDController

    plant = MassSpringDamper(
        mass=1.0,
        damping=10.0,
        spring_constant=20.0,
        initial_position=0.0,
        initial_velocity=0.0,
        external_force=0.0,
    )

    controller = PIDController(
        kp=200.0,
        ki=0.0,
        kd=0.0,
        setpoint=1.0,
    )

    results = run_simulation(
        plant=plant,
        controller=controller,
        duration=5.0,
        dt=0.001,
    )

    print(f"Final position: {results['position'][-1]:.4f} m")
    print(f"Final error: {results['error'][-1]:.4f} m")
    print(f"Final control force: {results['control_force'][-1]:.4f} N")

