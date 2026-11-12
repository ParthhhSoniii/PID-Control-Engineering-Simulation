"""
Experiment 02: P-Only Control

Objective:
Study the effect of proportional gain (Kp)
on a mass-spring-damper system.
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

KP_VALUES = [20, 50, 100, 200]

RESULTS_DIR = Path("results/02_p_only")


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
# RUN EXPERIMENT
# --------------------------------------------------

def run_p_only_experiment():

    experiment_results = {}

    print("\nP-ONLY CONTROL EXPERIMENT")
    print("=" * 60)

    for kp in KP_VALUES:

        print(f"\nRunning simulation: Kp = {kp}")

        # 1. Create a fresh plant
        plant = create_plant()

        # 2. Configure proportional controller
        controller = PIDController(
            kp=kp,
            ki=0.0,
            kd=0.0,
            setpoint=SETPOINT
        )

        # 3. Use the shared simulation engine
        results = run_simulation(
            plant=plant,
            controller=controller,
            duration=DURATION,
            dt=DT
        )

        # 4. Store simulation results
        experiment_results[kp] = results

        # 5. Calculate theoretical equilibrium
        theoretical_position = (
            kp / (kp + SPRING_CONSTANT)
        ) * SETPOINT

        theoretical_error = SETPOINT - theoretical_position

        final_position = results["position"][-1]
        actual_error = SETPOINT - final_position

        print(f"Final position       : {final_position:.4f} m")
        print(f"Theoretical position : {theoretical_position:.4f} m")
        print(f"Actual error         : {actual_error:.4f} m")
        print(f"Theoretical error    : {theoretical_error:.4f} m")

    # metrics = generate_comparison_report(
    #     experiments=experiment_results,
    #     setpoint=SETPOINT,
    #     output_dir="results/02_p_only"
    # )

    return experiment_results


# --------------------------------------------------
# GENERATE GRAPHS
# --------------------------------------------------

def generate_experiment_graphs(experiment_results):

        generate_comparison_report(
            experiments=experiment_results,
            setpoint=SETPOINT,
            output_dir="results/graphs/02_p_only"
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    results = run_p_only_experiment()

    generate_experiment_graphs(results)

    print("\n" + "=" * 60)
    print("P-ONLY EXPERIMENT COMPLETED")
    print("=" * 60)