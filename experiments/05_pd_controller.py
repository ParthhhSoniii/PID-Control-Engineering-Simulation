"""
Experiment 05: PD Controller — Derivative Gain Comparison

Run from the project root:
    python -m experiments.05_pd_controller

For every Kd value:
  1. A fresh, identical plant/controller pair is created.
  2. src.simulation.run_simulation() performs the closed-loop iteration.
  3. src.performance.calculate_performance() evaluates the response.
  4. src.performance.plot_pd_comparisons() creates six overlay graphs.

All graph-generation logic is kept in src/performance.py.
"""

from pathlib import Path

from src.plants.mass_spring_damper import MassSpringDamper
from src.pid_controller import PIDController
from src.simulation import run_simulation
from src.performance import calculate_performance, plot_pd_comparisons


# ---------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

MASS = 1.0
DAMPING = 10.0
SPRING_CONSTANT = 20.0

SETPOINT = 1.0
INITIAL_POSITION = 0.0
INITIAL_VELOCITY = 0.0

# Hold Kp fixed and vary only Kd. Ki=0 makes this a PD experiment.
KP = 100.0
KI = 0.0
KD_VALUES = [0.0, 5.0, 10.0, 20.0, 30.0]

DURATION = 5.0
DT = 0.001

OUTPUT_DIR = Path("results/05_pd_controller")


# ---------------------------------------------------------------------
# FACTORY FUNCTIONS
# ---------------------------------------------------------------------

def create_plant():
    """Create a fresh plant with identical initial conditions."""
    return MassSpringDamper(
        mass=MASS,
        damping=DAMPING,
        spring_constant=SPRING_CONSTANT,
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        external_force=0.0,
    )


def create_controller(kd):
    """Create a PD controller for the selected derivative gain."""
    return PIDController(
        kp=KP,
        ki=KI,
        kd=kd,
        setpoint=SETPOINT,
    )


def print_metric_summary(metrics_by_gain):
    """Print the metrics calculated by performance.py."""
    print("\n" + "-" * 104)
    print(
        f"{'Kd':>8} | {'Rise time (s)':>15} | {'Overshoot (%)':>15} | "
        f"{'Settling 2% (s)':>18} | {'Steady-state error (m)':>23}"
    )
    print("-" * 104)

    for kd, metrics in metrics_by_gain.items():
        rise = metrics.get("rise_time", float("nan"))
        overshoot = metrics.get("overshoot_percent", float("nan"))
        settling = metrics.get("settling_time_2_percent", float("nan"))
        sse = metrics.get("steady_state_error", float("nan"))

        print(
            f"{kd:8.2f} | {rise:15.4f} | {overshoot:15.4f} | "
            f"{settling:18.4f} | {sse:23.6f}"
        )

    print("-" * 104)


# ---------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_results = {}
    metrics_by_gain = {}

    print("\n" + "=" * 78)
    print("EXPERIMENT 05: PD CONTROLLER — Kd OVERLAY COMPARISON")
    print("=" * 78)

    print(
        f"Plant: m={MASS} kg, c={DAMPING} N.s/m, "
        f"k={SPRING_CONSTANT} N/m"
    )
    print(f"Setpoint: {SETPOINT:.2f} m")
    print(f"Fixed gains: Kp={KP:.2f}, Ki={KI:.2f}")
    print(f"Kd values: {KD_VALUES}")
    print(f"Duration={DURATION:.2f} s, dt={DT:.4f} s")

    for kd in KD_VALUES:
        label = f"Kd = {kd:g}"
        print(f"\nRunning {label} ...")

        plant = create_plant()
        controller = create_controller(kd)

        # Shared simulation engine: no plant or PID equations are duplicated.
        results = run_simulation(
            plant=plant,
            controller=controller,
            duration=DURATION,
            dt=DT,
        )

        experiment_results[label] = results

        # Evaluate each Kd run using the shared performance module.
        metrics_by_gain[kd] = calculate_performance(
            results=results,
            setpoint=SETPOINT,
        )

        m = metrics_by_gain[kd]

        print(
            f"Completed {label}: rise={m['rise_time']:.4f} s, "
            f"overshoot={m['overshoot_percent']:.3f}%, "
            f"settling={m['settling_time_2_percent']:.4f} s, "
            f"SSE={m['steady_state_error']:.6f} m"
        )

    # All six comparison figures are generated inside performance.py.
    comparison_metrics = plot_pd_comparisons(
        experiments=experiment_results,
        setpoint=SETPOINT,
        output_dir=OUTPUT_DIR,
    )

    print_metric_summary(metrics_by_gain)
    
    expected_graphs = [
        "position_comparison.png",
        "velocity_comparison.png",
        "acceleration_comparison.png",
        "control_force_comparison.png",
        "error_comparison.png",
        "phase_portrait_comparison.png",
    ]

    print("\n" + "=" * 78)
    print("PD EXPERIMENT COMPLETE")
    print("=" * 78)

    print(f"Graphs saved in: {OUTPUT_DIR}")

    for graph_name in expected_graphs:
        print(f"  - {OUTPUT_DIR / graph_name}")

    print("\nEngineering interpretation:")
    print(
        "- Kd=0 is the P-only reference response.\n"
        "- Increasing Kd adds rate-sensitive correction and may reduce "
        "overshoot and oscillation.\n"
        "- Excessive Kd can slow the response and amplify measurement noise.\n"
        "- Ki remains zero, so PD does not eliminate the spring-load "
        "steady-state offset.\n"
        "- Compare all six plots together: position, velocity, acceleration, "
        "applied force, error and phase portrait."
    )


if __name__ == "__main__":
    main()