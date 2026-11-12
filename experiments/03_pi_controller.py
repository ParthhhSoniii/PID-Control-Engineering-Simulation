"""
Experiment 03: PI Control

Objective:
Study the effect of integral gain (Ki) on the
response of a mass-spring-damper system.

The proportional gain remains constant while
integral gain is varied.
"""

from pathlib import Path

from src.plants.mass_spring_damper import MassSpringDamper
from src.pid_controller import PIDController
from src.simulation import run_simulation

from src.performance import (
    plot_position_response,
    plot_velocity_response,
    plot_acceleration_response,
    plot_control_force,
    plot_error_response,
    plot_phase_portrait,
    generate_comparison_report
)

# --------------------------------------------------
# EXPERIMENT CONFIGURATION
# --------------------------------------------------

MASS = 1.0
DAMPING = 10.0
SPRING_CONSTANT = 20.0

INITIAL_POSITION = 0.0
INITIAL_VELOCITY = 0.0

SETPOINT = 1.0

DURATION = 5.0
DT = 0.001

# Keep Kp constant and vary Ki
KP = 100.0

KI_VALUES = [ 25, 50, 100 , 150]

RESULTS_DIR = Path("results/03_pi_control")


# --------------------------------------------------
# PLANT CREATION
# --------------------------------------------------

def create_plant():

    return MassSpringDamper(
        mass=MASS,
        damping=DAMPING,
        spring_constant=SPRING_CONSTANT,
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        external_force=0.0
    )


# --------------------------------------------------
# RUN PI EXPERIMENT
# --------------------------------------------------

def run_pi_experiment():

    experiment_results = {}

    print("\nPI CONTROL EXPERIMENT")
    print("=" * 60)

    for ki in KI_VALUES:

        print(f"\nRunning simulation: Kp = {KP}, Ki = {ki}")

        # 1. Create a fresh plant
        plant = create_plant()

        # 2. Configure PI controller
        controller = PIDController(
            kp=KP,
            ki=ki,
            kd=0.0,
            setpoint=SETPOINT
        )

        # 3. Execute shared simulation engine
        results = run_simulation(
            plant=plant,
            controller=controller,
            duration=DURATION,
            dt=DT
        )

        # 4. Store results
        experiment_results[ki] = results

        # 5. Calculate final error
        final_position = results["position"][-1]
        actual_error = SETPOINT - final_position

        print(f"Final position : {final_position:.4f} m")
        print(f"Final error    : {actual_error:.4f} m")

    return experiment_results


# --------------------------------------------------
# GENERATE GRAPHS
# --------------------------------------------------

def generate_experiment_graphs(experiment_results):
    
    generate_comparison_report(
        experiments=experiment_results,
        setpoint=SETPOINT,
        output_dir="results/03_pi_control"
    )

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    results = run_pi_experiment()

    generate_experiment_graphs(results)

    print("\n" + "=" * 60)
    print("PI CONTROL EXPERIMENT COMPLETED")
    print("=" * 60)