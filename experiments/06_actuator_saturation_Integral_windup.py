import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.plants.mass_spring_damper import MassSpringDamper
from src.pid_controller import PIDController
from src.simulation import run_simulation


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MASS = 1.0
DAMPING = 10.0
SPRING_CONSTANT = 20.0

SETPOINT = 1.0

KP = 100.0
KI = 100.0
KD = 0.0

ACTUATOR_LIMITS = (-30.0, 30.0)

DURATION = 5.0
DT = 0.001

RESULTS_DIR = Path("results/06_actuator_saturation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER: CREATE A FRESH PLANT
# ============================================================

def create_plant():

    return MassSpringDamper(
        mass=MASS,
        damping=DAMPING,
        spring_constant=SPRING_CONSTANT,
        initial_position=0.0,
        initial_velocity=0.0,
        external_force=0.0
    )


# ============================================================
# SCENARIO A: NO ACTUATOR SATURATION
# ============================================================

def run_scenario_a():

    print("\nRunning Scenario A: No Actuator Saturation. Simple PI Controller.")

    plant = create_plant()

    controller = PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        setpoint=SETPOINT,
        output_limits=None,
        anti_windup=False
    )

    results = run_simulation(
        plant=plant,
        controller=controller,
        duration=DURATION,
        dt=DT
    )

    return results


# ============================================================
# SCENARIO B: SATURATION WITHOUT ANTI-WINDUP
# ============================================================

def run_scenario_b():

    print("\nRunning Scenario B: Saturation Without Anti-Windup")

    plant = create_plant()

    controller = PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        setpoint=SETPOINT,
        output_limits=ACTUATOR_LIMITS,
        anti_windup=False
    )

    results = run_simulation(
        plant=plant,
        controller=controller,
        duration=DURATION,
        dt=DT
    )

    return results


# ============================================================
# SCENARIO C: SATURATION WITH ANTI-WINDUP
# ============================================================

def run_scenario_c():

    print("\nRunning Scenario C: Saturation With Anti-Windup")

    plant = create_plant()

    controller = PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        setpoint=SETPOINT,
        output_limits=ACTUATOR_LIMITS,
        anti_windup=True
    )

    results = run_simulation(
        plant=plant,
        controller=controller,
        duration=DURATION,
        dt=DT
    )

    return results


# ============================================================
# GRAPH 1: POSITION RESPONSE COMPARISON
# ============================================================

def plot_position_comparison(experiments):

    plt.figure(figsize=(11, 6))

    for label, results in experiments.items():

        plt.plot(
            results["time"],
            results["position"],
            label=label,
            linewidth=1.5
        )

    plt.axhline(
        SETPOINT,
        linestyle="--",
        color="black",
        label="Target Position"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Position (m)")
    plt.title("Experiment 06: Position Response Comparison")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "position_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 2: RAW FORCE VS ACTUALLY APPLIED FORCE
# ============================================================

def plot_control_force_comparison(experiments):

    plt.figure(figsize=(12, 6))

    for label, results in experiments.items():

        # Raw force requested by the controller
        raw_line, = plt.plot(
            results["time"],
            results["raw_control_force"],
            linestyle="--",
            linewidth=1,
            label=f"{label} - Requested"
        )

        # Actual force applied to the plant
        plt.plot(
            results["time"],
            results["control_force"],
            color=raw_line.get_color(),
            linewidth=1.5,
            label=f"{label} - Applied"
        )

    plt.axhline(
        ACTUATOR_LIMITS[0],
        color="black",
        linestyle=":",
        label="Actuator Limits"
    )

    plt.axhline(
        ACTUATOR_LIMITS[1],
        color="black",
        linestyle=":"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Force (N)")
    plt.title("Requested Force vs Actual Applied Force")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "control_force_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 3: INTEGRAL CONTRIBUTION COMPARISON
# ============================================================

def plot_integral_comparison(experiments):

    plt.figure(figsize=(11, 6))

    for label, results in experiments.items():

        plt.plot(
            results["time"],
            results["integral_term"],
            label=label,
            linewidth=1.5
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Integral Contribution (N)")
    plt.title("Integral Windup: Comparison of Three Scenarios")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "integral_contribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def print_summary(experiments):

    print("\n" + "=" * 65)
    print("EXPERIMENT 06: PERFORMANCE SUMMARY")
    print("=" * 65)

    for label, results in experiments.items():

        final_position = results["position"][-1]

        final_error = SETPOINT - final_position

        maximum_requested_force = max(
            abs(force)
            for force in results["raw_control_force"]
        )

        final_integral = results["integral_term"][-1]

        saturation_steps = sum(
            abs(raw - applied) > 1e-9
            for raw, applied in zip(
                results["raw_control_force"],
                results["control_force"]
            )
        )

        print(f"\n{label}")
        print("-" * 45)

        print(f"Final Position          : {final_position:.4f} m")
        print(f"Final Error             : {final_error:.4f} m")
        print(f"Maximum Requested Force : {maximum_requested_force:.2f} N")
        print(f"Final Integral Term     : {final_integral:.2f} N")
        print(f"Saturated Time Steps    : {saturation_steps}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("EXPERIMENT 06: ACTUATOR SATURATION AND INTEGRAL WINDUP")
    print("=" * 65)

    # Run each scenario independently
    scenario_a_results = run_scenario_a()
    scenario_b_results = run_scenario_b()
    scenario_c_results = run_scenario_c()

    # Store results for comparison
    experiments = {
        "A: No Saturation": scenario_a_results,
        "B: Saturation Without Anti-Windup": scenario_b_results,
        "C: Saturation With Anti-Windup": scenario_c_results
    }

    # Generate comparison graphs
    plot_position_comparison(experiments)
    plot_control_force_comparison(experiments)
    plot_integral_comparison(experiments)

    # Print numerical summary
    print_summary(experiments)

    print("\nGraphs saved in:", RESULTS_DIR)