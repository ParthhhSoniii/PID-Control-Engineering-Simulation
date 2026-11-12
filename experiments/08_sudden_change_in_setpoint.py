"""
Experiment 08: Derivative Kick
==============================

Scenario A only:
A sudden setpoint change causes a spike in the derivative term
when the PID derivative is calculated from the error.

Run from the project root:
    python -m experiments.08_derivative_kick
"""

from pathlib import Path
import matplotlib.pyplot as plt

from src.plants.mass_spring_damper import MassSpringDamper
from src.pid_controller import PIDController


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MASS = 1.0
DAMPING = 10.0
SPRING_CONSTANT = 20.0

INITIAL_SETPOINT = 0.5
FINAL_SETPOINT = 1.0
SETPOINT_CHANGE_TIME = 1.0

KP = 50.0
KI = 20.0
KD = 5.0

DURATION = 4.0
DT = 0.001

RESULTS_DIR = Path("results/08_sudden_setpoint_change")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE PLANT AND CONTROLLER
# ============================================================

plant = MassSpringDamper(
    mass=MASS,
    damping=DAMPING,
    spring_constant=SPRING_CONSTANT,
    initial_position=0.0,
    initial_velocity=0.0,
    external_force=0.0,
)

controller = PIDController(
    kp=KP,
    ki=KI,
    kd=KD,
    setpoint=INITIAL_SETPOINT,
)

controller.reset()

# Initialize previous_error to the starting error.
# This prevents an unrelated derivative spike at t = 0.
controller.previous_error = INITIAL_SETPOINT - plant.position


# ============================================================
# SIMULATION
# ============================================================

time_values = []
position_values = []
setpoint_values = []
error_values = []
control_force_values = []
derivative_values = []

number_of_steps = int(round(DURATION / DT))

for step in range(number_of_steps + 1):

    current_time = step * DT

    # At t = 1 second, suddenly change the target from 0.5 m to 1.0 m.
    if current_time < SETPOINT_CHANGE_TIME:
        controller.setpoint = INITIAL_SETPOINT
    else:
        controller.setpoint = FINAL_SETPOINT

    # Read actual plant position (no sensor noise in this experiment).
    current_position = plant.position

    # Calculate PID force using conventional derivative-on-error.
    force = controller.calculate_output(
        current_position=current_position,
        dt=DT,
    )

    # Apply force to the plant.
    plant.external_force = force

    # Record signals.
    time_values.append(current_time)
    position_values.append(plant.position)
    setpoint_values.append(controller.setpoint)
    error_values.append(controller.error)
    control_force_values.append(force)
    derivative_values.append(controller.derivative_term)

    # Advance plant physics.
    if step < number_of_steps:
        plant.update_state(DT)


# ============================================================
# GRAPH 1: POSITION RESPONSE AND SETPOINT CHANGE
# ============================================================

plt.figure(figsize=(11, 5))

plt.plot(
    time_values,
    position_values,
    label="Actual Plant Position",
)

plt.plot(
    time_values,
    setpoint_values,
    linestyle="--",
    label="Setpoint",
)

plt.axvline(
    SETPOINT_CHANGE_TIME,
    linestyle=":",
    label="Sudden Setpoint Change",
)

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Experiment 08: Position Response to a Sudden Setpoint Change")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "position_response.png",
    dpi=300,
)
plt.close()


# ============================================================
# GRAPH 2: DERIVATIVE KICK — FULL SCALE + ZOOMED VIEW
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True
)

# Full-scale graph
ax1.plot(
    time_values,
    derivative_values,
    label="Derivative Contribution"
)

ax1.axvline(
    SETPOINT_CHANGE_TIME,
    linestyle=":",
    label="Setpoint Change"
)

ax1.set_ylabel("Derivative Force (N)")
ax1.set_title("Derivative Kick: Full Response")
ax1.grid(True)
ax1.legend()

# Zoomed graph
ax2.plot(
    time_values,
    derivative_values,
    label="Derivative Contribution"
)

ax2.axvline(
    SETPOINT_CHANGE_TIME,
    linestyle=":",
    label="Setpoint Change"
)

ax2.set_ylim(-20, 20)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Derivative Force (N)")
ax2.set_title("Zoomed View: Normal Derivative Behaviour")
ax2.grid(True)
ax2.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "derivative_kick.png",
    dpi=300
)

plt.close()

# ============================================================
# SUMMARY
# ============================================================

maximum_derivative = max(abs(value) for value in derivative_values)
maximum_force = max(abs(value) for value in control_force_values)

print("=" * 68)
print("EXPERIMENT 08: DERIVATIVE KICK")
print("=" * 68)
print(f"Initial setpoint          : {INITIAL_SETPOINT:.2f} m")
print(f"Final setpoint            : {FINAL_SETPOINT:.2f} m")
print(f"Setpoint change time      : {SETPOINT_CHANGE_TIME:.2f} s")
print(f"Maximum derivative output : {maximum_derivative:.2f} N")
print(f"Maximum absolute force    : {maximum_force:.2f} N")
print(f"Graphs saved in           : {RESULTS_DIR}")
