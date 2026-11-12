"""
Experiment 07: Position Sensor Noise
====================================

Scenario A: Ideal position sensor (no measurement noise)
Scenario B: Noisy position sensor (uniform noise in +/- 1 mm)

Run from the project root:
    python -m experiments.07_position_sensor

Assumed interfaces:
- PositionSensor(noise_limit=0.001, seed=42)
- run_simulation(plant, controller, duration, dt, sensor=None)
- Simulation results contain:
      time, position, measured_position, error, control_force

Here, "position" is the actual plant position and "measured_position"
is the value delivered by the sensor to the controller.
If your simulation.py uses a different keyword or result-key name,
change the adapter lines below.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.plants.mass_spring_damper import MassSpringDamper
from src.pid_controller import PIDController
from src.simulation import run_simulation
from src.sensors.position_sensor import PositionSensor

from src.performance import (
    plot_position_comparison,
    plot_control_force_comparison,
    plot_error_comparison,
)

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MASS = 1.0
DAMPING = 10.0
SPRING_CONSTANT = 20.0
SETPOINT = 1.0

# Keep gains identical in both scenarios.
KP = 50.0
KI = 100.0
KD = 5.0

NOISE_LIMIT = 0.1       # +/- 10 cm in metres
RANDOM_SEED = 42

DURATION = 5.0
DT = 0.001

RESULTS_DIR = Path("results/07_position_sensor")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def create_plant():
    return MassSpringDamper(
        mass=MASS,
        damping=DAMPING,
        spring_constant=SPRING_CONSTANT,
        initial_position=0.0,
        initial_velocity=0.0,
        external_force=0.0,
    )


def create_controller():
    return PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        setpoint=SETPOINT,
    )


def get_measured_position(results):
    """Adapt this key if simulation.py uses a different result name."""
    if "position" in results:
        return results["position"]

    raise KeyError(
        "Simulation results need 'measured_position' (or 'sensor_reading'). "
        "Check the return dictionary in src/simulation.py."
    )


# ============================================================
# SCENARIO A: IDEAL SENSOR
# ============================================================

def run_scenario_a():
    print("\nRunning Scenario A: Ideal Position Sensor")

    plant = create_plant()
    controller = create_controller()

    results = run_simulation(
        plant=plant,
        controller=controller,
        duration=DURATION,
        dt=DT,
        position_sensor_obj=False,
    )

    # Make the ideal measurement explicit for consistent comparisons.
    results["measured_position"] = list(results["position"])
    return results


# ============================================================
# SCENARIO B: NOISY SENSOR (+/- 1 mm)
# ============================================================

def run_scenario_b():
    print("\nRunning Scenario B: Noisy Position Sensor (+/- 1 mm)")

    plant = create_plant()
    controller = create_controller()

    sensor = PositionSensor(
        noise_limit=NOISE_LIMIT,
        seed=RANDOM_SEED,
    )

    results = run_simulation(
        plant=plant,
        controller=controller,
        duration=DURATION,
        dt=DT,
        position_sensor_obj=sensor
    )

    get_measured_position(results)
    return results


# ============================================================
# GRAPH 1: ACTUAL POSITION VS MEASURED POSITION
# ============================================================

def plot_sensor_measurements(noisy_results):
    time = noisy_results["time"]
    actual = noisy_results["position"]
    measured = get_measured_position(noisy_results)

    plt.figure(figsize=(12, 6))
    plt.plot(time, actual, label="Actual Plant Position", linewidth=2)
    plt.plot(
        time,
        measured,
        label="Noisy Sensor Measurement",
        linewidth=0.8,
        alpha=0.8,
    )
    plt.axhline(SETPOINT, linestyle="--", label="Target Position")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Actual Position vs Noisy Sensor Reading")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_measured_position.png", dpi=300)
    plt.close()


# ============================================================
# GRAPH 2: SENSOR NOISE SIGNAL
# ============================================================

def plot_sensor_noise(results):

    time = results["time"]

    actual = np.array(results["position"])
    measured = np.array(results["measured_position"])

    noise = (measured - actual) * 1000  # Convert to mm

    plt.figure(figsize=(12, 5))

    plt.plot(
        time,
        noise,
        label="Sensor Measurement Noise",
        linewidth=0.8
    )

    plt.axhline(1, linestyle="--", label="+1 mm limit")
    plt.axhline(-1, linestyle="--", label="-1 mm limit")
    plt.axhline(0, color="black", linewidth=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("Measurement Noise (mm)")
    plt.title("Position Sensor Noise Over Time")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "sensor_noise_signal.png",
        dpi=300
    )

    plt.close()

# ============================================================
# GRAPH 3: CONTROLLER ERROR COMPARISON
# ============================================================

def plot_tracking_error_comparison(experiments):
    plt.figure(figsize=(12, 5))

    for label, results in experiments.items():
        plt.plot(
            results["time"],
            results["error"],
            label=label,
            linewidth=1,
        )

    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Controller Feedback Error (m)")
    plt.title("Controller Error: Ideal vs Noisy Sensor")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tracking_error_comparison.png", dpi=300)
    plt.close()


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def print_summary(experiments):
    print("\n" + "=" * 68)
    print("EXPERIMENT 07: SENSOR NOISE PERFORMANCE SUMMARY")
    print("=" * 68)

    for label, results in experiments.items():
        final_position = results["position"][-1]
        final_error = SETPOINT - final_position

        forces = results["control_force"]
        measured = results["measured_position"]
        actual = results["position"]

        noise_values = np.array(measured) - np.array(actual)

        max_abs_noise = np.max(np.abs(noise_values))

        noise_values = [
            measured_value - actual_value
            for measured_value, actual_value in zip(measured, actual)
        ]

        max_abs_noise = max(abs(value) for value in noise_values)

        print(f"\n{label}")
        print("-" * 48)
        print(f"Final actual position : {final_position:.5f} m")
        print(f"Final position error  : {final_error:.5f} m")
        print(f"Minimum control force : {min(forces):.3f} N")
        print(f"Maximum control force : {max(forces):.3f} N")
        print(f"Maximum sensor noise  : {max_abs_noise:.6f} m")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 68)
    print("EXPERIMENT 07: POSITION SENSOR NOISE")
    print("=" * 68)
    print(f"Setpoint: {SETPOINT:.2f} m")
    print(f"PID gains: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"Sensor noise: +/- {NOISE_LIMIT:.3f} m")

    scenario_a_results = run_scenario_a()
    scenario_b_results = run_scenario_b()

    experiments = {
        "A: Ideal Sensor": scenario_a_results,
        "B: Noisy Sensor (+/- 1 mm)": scenario_b_results,
    }

    # Sensor-specific graphs.
    plot_sensor_measurements(scenario_b_results)
    plot_sensor_noise(scenario_b_results)
    plot_tracking_error_comparison(experiments)

    # Reuse relevant comparison graphs from performance.py.
    plot_position_comparison(
        experiments=experiments,
        setpoint=SETPOINT,
        output_dir=RESULTS_DIR,
        filename="position_response_comparison.png",
    )
    plot_control_force_comparison(
        experiments=experiments,
        output_dir=RESULTS_DIR,
    )
    plot_error_comparison(
        experiments=experiments,
        output_dir=RESULTS_DIR,
    )

    print_summary(experiments)
    print("\nGraphs saved in:", RESULTS_DIR)
