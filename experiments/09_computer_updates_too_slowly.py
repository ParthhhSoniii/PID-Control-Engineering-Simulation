from pathlib import Path
import matplotlib.pyplot as plt
from src.plants.mass_spring_damper import MassSpringDamper
from src.pid_controller import PIDController

# Experiment 09: Slow Controller Update Frequency
MASS, DAMPING, SPRING = 1.0, 10.0, 20.0
SETPOINT = 1.0
KP, KI, KD = 100.0, 100.0, 4.0
DURATION = 5.0

# Plant physics remains at 1 ms. Only controller sampling changes.
DT = 0.001
CONTROLLER_PERIODS = [0.001, 0.01, 0.05, 0.1]  # 1000, 100, 20, 10 Hz

RESULTS_DIR = Path("results/09_slow_update_frequency")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_frequency_case(controller_period):
    plant = MassSpringDamper(
        mass=MASS, damping=DAMPING, spring_constant=SPRING,
        initial_position=0.0, initial_velocity=0.0, external_force=0.0
    )
    controller = PIDController(
        kp=KP, ki=KI, kd=KD, setpoint=SETPOINT,
    )
    controller.reset()

    total_steps = int(round(DURATION / DT))

    update_every = max(1, int(round(controller_period / DT)))

    held_force = 0.0
    data = {"time": [], "position": [], "force": []}

    for step in range(total_steps + 1):
        t = step * DT

        # Controller samples position and recalculates force only at these instants.
        if step % update_every == 0:
            held_force = controller.calculate_output(
                current_position=plant.position,
                dt=controller_period
            )

        # Zero-order hold: actuator keeps the latest command until next update.
        plant.external_force = held_force
        data["time"].append(t)
        data["position"].append(plant.position)
        data["force"].append(held_force)

        if step < total_steps:
            plant.update_state(DT)

    return data


if __name__ == "__main__":
    experiments = {}
    for period in CONTROLLER_PERIODS:
        frequency = 1.0 / period
        label = f"{frequency:g} Hz (Ts={period:g}s)"
        print("Running", label)
        experiments[label] = run_frequency_case(period)

    plt.figure(figsize=(11, 5))
    for label, result in experiments.items():
        plt.plot(result["time"], result["position"], label=label)
    plt.axhline(SETPOINT, linestyle="--", label="Target")
    plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
    plt.title("Experiment 09: Effect of Controller Update Frequency")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "position_response_comparison.png", dpi=300); plt.close()

    plt.figure(figsize=(11, 5))
    for label, result in experiments.items():
        plt.plot(result["time"], result["force"], label=label)
    plt.xlabel("Time (s)"); plt.ylabel("Held control force (N)")
    plt.title("Experiment 09: Control Force at Different Update Frequencies")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "control_force_comparison.png", dpi=300); plt.close()

    print("Graphs saved in:", RESULTS_DIR)
