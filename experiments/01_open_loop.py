'''
What this experiment does
- Simulates the 1 kg mass-spring-damper plant without a PID controller.
- Applies a constant external force of 20 N.
- Uses your existing MassSpringDamper class.
- Generates three graphs:
  - Position response
  - Velocity response
  - Acceleration response
- Compares the final position against the theoretical equilibrium:
'''

"""
01_open_loop.py

Open-loop experiment using the shared simulation engine in src/simulation.py.

A constant force is applied directly to the mass-spring-damper plant.
There is no feedback: the input does not depend on the measured position.

Run from the project root:
    python -m experiments.01_open_loop
"""

"""
01_open_loop.py

Open-loop experiment using the shared simulation engine in src/simulation.py.

A constant force is applied directly to the mass-spring-damper plant.
There is no feedback: the input does not depend on the measured position.

Run from the project root:
    python -m experiments.01_open_loop
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.plants.mass_spring_damper import MassSpringDamper
from src.simulation import run_simulation

from src.performance import (
    plot_position_response,
    plot_velocity_response,
    plot_acceleration_response,
    plot_control_force,
    plot_error_response,
    plot_phase_portrait
)


# ---------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

MASS = 1.0                  # kg
DAMPING = 10.0              # N.s/m
SPRING_CONSTANT = 20.0      # N/m

INITIAL_POSITION = 0.0      # m
INITIAL_VELOCITY = 0.0      # m/s

APPLIED_FORCE = 20.0        # N, constant open-loop input

DURATION = 5.0              # seconds
DT = 0.001                  # seconds

REFERENCE_POSITION = 1.0    # m, shown for comparison only

RESULTS_DIR = Path("results/open_loop")


# ---------------------------------------------------------------------
# CONSTANT OPEN-LOOP INPUT
# ---------------------------------------------------------------------

class ConstantForceInput:
    """
    Small input source compatible with run_simulation().

    Unlike a PID controller, this object does not use position or error.
    It returns the same force at every simulation step.
    """

    def __init__(self, force):
        self.force = float(force)
        self.error = 0.0


    def reset(self):
        """Included to satisfy the simulation engine's controller interface."""
        self.error = 0.0

    def calculate_output(self, current_position, dt):
        """Return a fixed force, independent of the measured position."""
        return self.force


# ---------------------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------------------

def run_open_loop_experiment():
    """Create the plant and delegate all time stepping to run_simulation()."""

    plant = MassSpringDamper(
        mass=MASS,
        damping=DAMPING,
        spring_constant=SPRING_CONSTANT,
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        external_force=0.0,
    )

    open_loop_input = ConstantForceInput(APPLIED_FORCE)

    # The shared simulation module handles the simulation loop,
    # state updates, and recording of time/position/velocity/etc.
    results = run_simulation(
        plant=plant,
        controller=open_loop_input,
        duration=DURATION,
        dt=DT,
    )

    # Generate all experiment graphs
    plot_position_response(results, setpoint=1.0, output_dir=RESULTS_DIR)
    plot_velocity_response(results, output_dir=RESULTS_DIR)
    plot_acceleration_response(results, output_dir=RESULTS_DIR)
    plot_control_force(results, output_dir=RESULTS_DIR)
    plot_error_response(results, output_dir=RESULTS_DIR)
    plot_phase_portrait(results, output_dir=RESULTS_DIR)

    return results

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    results = run_open_loop_experiment()
    # plot_open_loop_response(results)

    theoretical_equilibrium = APPLIED_FORCE / SPRING_CONSTANT

    print("\nOPEN-LOOP EXPERIMENT")
    print("-" * 48)
    print(f"Applied constant force  : {APPLIED_FORCE:.2f} N")
    print(f"Theoretical equilibrium : {theoretical_equilibrium:.4f} m")
    print(f"Final simulated position: {results['position'][-1]:.4f} m")
    print(f"Final simulated velocity: {results['velocity'][-1]:.6f} m/s")
    print(f"Simulation duration     : {DURATION:.2f} s")
    print(f"Time step               : {DT:.4f} s")
    print(f"Graphs saved in         : {RESULTS_DIR}")
    print("-" * 48)
