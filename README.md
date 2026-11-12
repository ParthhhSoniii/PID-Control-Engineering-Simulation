# PID Control — Modern Monk Labs Internship Assignment

A from-scratch Python implementation and experiment suite for studying PID control on a
mass–spring–damper plant. The project uses **NumPy** and **Matplotlib** and implements
the PID logic directly rather than using a ready-made control library.

## Project objectives

- Explain closed-loop feedback and the P, I and D terms.
- Simulate a 1 kg mass–spring–damper system with `c = 10 N·s/m`, `k = 20 N/m`, and a 1 m target.
- Compare P, PI, PD and PID controller behaviour.
- Evaluate rise time, overshoot, settling time and steady-state error.
- Investigate actuator saturation and integral windup, sensor noise, derivative kick and slow controller updates.
- Demonstrate manual tuning and Ziegler–Nichols tuning on `G(s) = 1/(s+1)^3`.
- Recreate the figures using the experiment scripts and shared plotting module.

## Project structure

```text
Modern_MonkLabs_Assignment/
├── README.md
├── requirements.txt
├── app.py                         # Optional Flask API for the interactive dashboard
├── web/
│   └── index.html                 # Browser dashboard; calls the Flask simulation API
├── src/
│   ├── pid_controller.py          # Reusable PID controller logic
│   ├── simulation.py              # Closed-loop simulation coordinator
│   ├── performance.py             # Performance metrics and standard plots
│   ├── plants/
│   │   ├── mass_spring_damper.py  # Main physical plant model
│   │   └── unknown_transfer_function.py  # Third-order plant for ZN experiments
│   └── sensors/
│       └── position_sensor.py    # Position measurement with configurable noise
├── experiments/
│   ├── 01_open_loop.py
│   ├── 02_p_only.py
│   ├── 03_pi_controller.py
│   ├── 04_pid_controller.py
│   ├── 05_pd_controller.py
│   ├── 06_actuator_saturation.py
│   ├── 07_position_sensor.py
│   ├── 08_derivative_kick.py
│   ├── 09_slow_update_frequency.py
│   └── 11_ziegler_nichols.py
├── results/
│   └── graphs/
│       ├── 01_open_loop/
│       ├── 02_p_only/
│       ├── 03_pi_control/
│       ├── 04_pid_control/
│       ├── 06_actuator_saturation/
│       ├── 07_position_sensor/
│       ├── 08_sudden_setpoint_change/
│       └── 09_slow_update_frequency/
└── PID_Control_Internship_Report.pdf
```

> This is the intended project tree based on the files and structure discussed during
> development. Some output folders are created only after running their experiment.
> Keep the names in your local checkout if they differ from this illustrative tree.

## Architecture

```text
Setpoint
   |
   v
PIDController <----- measured position <----- PositionSensor (optional)
   |                                            ^
   | force command                              | actual position
   v                                            |
MassSpringDamper plant -------------------------+
   |
   v
simulation.py records time, state, error and force
   |
   v
performance.py calculates metrics and saves plots
```

Responsibilities are separated: the plant owns the physics, the controller owns the PID
law, the simulation coordinates the feedback loop, and the performance module analyses
the results. Experiment scripts select parameters and call these shared components.

## Important source files: inputs and outputs

### `src/pid_controller.py`

**Purpose:** implements the PID law directly in Python.

```python
controller = PIDController(kp=100.0, ki=0.0, kd=10.0, setpoint=1.0)

force = controller.calculate_output(
    current_position=measured_position,
    dt=0.001,
)
```

**Inputs**

| Input | Meaning | Unit |
|---|---|---|
| `kp`, `ki`, `kd` | Proportional, integral and derivative gains | Depends on output/input units |
| `setpoint` | Desired position | m |
| `current_position` | Measured position | m |
| `dt` | Time since previous controller update | s |

**Output:** a force command in newtons. The controller also stores its latest `error`,
`proportional`, `integral_term`, `derivative`, `derivative_term` and `output` values for
inspection and plotting. `reset()` clears integral and previous-error history.

Discrete calculation:

```text
error[k]       = setpoint - measured_position[k]
integral[k]    = integral[k-1] + error[k] * dt
derivative[k]  = (error[k] - error[k-1]) / dt
force[k]       = kp*error[k] + ki*integral[k] + kd*derivative[k]
```

The first derivative sample is zero because there is no previous error. This prevents an
artificial startup spike. A later setpoint step can still cause derivative kick when D is
calculated from error.

### `src/simulation.py`

**Purpose:** coordinates the controller–plant feedback loop. It does not contain the PID
equation or the plant's physical equation.

Function signature:

```python
run_simulation(
    plant,
    controller,
    duration,
    dt,
    position_sensor_obj=None,
)
```

**Inputs**

- `plant`: initialized physical plant object.
- `controller`: initialized PID controller.
- `duration`: total simulation time in seconds.
- `dt`: simulation/controller step in seconds for the standard loop.
- `position_sensor_obj`: optional object with a `.read(actual_position)` method.

**Loop responsibilities**

1. Read the actual plant position.
2. Obtain a sensor measurement (or use actual position).
3. Pass the measurement and `dt` to `controller.calculate_output()`.
4. Assign the returned force to `plant.external_force`.
5. Record actual and measured positions, velocity, acceleration, error, applied force,
   raw force and P/I/D contributions.
6. Advance the plant by one time step.

**Returned dictionary**

```python
{
    "time": [...],
    "position": [...],             # actual plant position (m)
    "measured_position": [...],    # sensor output (m)
    "velocity": [...],             # m/s
    "acceleration": [...],         # m/s^2
    "error": [...],                # m
    "control_force": [...],        # applied force (N)
    "raw_control_force": [...],    # requested force before clipping, if available
    "proportional": [...],         # P contribution (N)
    "integral_term": [...],        # I contribution (N)
    "derivative_term": [...],      # D contribution (N)
}
```

Sensor noise affects the measurement supplied to the controller, not the plant's true
position. The actual and measured signals are retained separately.

### `src/performance.py`

**Purpose:** owns the project's performance calculations and graph generation. Experiment
scripts should call this module rather than duplicating plotting code.

Main analysis functions:

- `calculate_rise_time(time, position, setpoint)` — 10–90% rise time and crossing times.
- `calculate_overshoot(position, setpoint)` — peak overshoot as a percentage of step size.
- `calculate_settling_time(time, position, setpoint, tolerance=0.02)` — settling time for a
  selected tolerance band; `NaN` means the response did not remain in the band before the run ended.
- `calculate_steady_state_error(position, setpoint)` — target minus mean position over the final 5% of samples.
- `calculate_performance(results, setpoint)` — returns the four metrics and crossing times.

Plot/report functions include:

- `plot_position_response(...)`
- `plot_velocity_response(...)`
- `plot_acceleration_response(...)`
- `plot_control_force(...)`
- `plot_error_response(...)`
- `plot_pid_components(...)`
- `plot_phase_portrait(...)`
- `plot_controller_comparison(...)`
- `generate_performance_report(...)`

Example:

```python
from src.performance import generate_performance_report

metrics = generate_performance_report(
    results=simulation_results,
    setpoint=1.0,
    output_dir="results/graphs/experiment_name",
)
```

This saves the standard plots and returns the calculated performance metrics. Use the
exact function names present in your local `performance.py` if your checkout has a different
revision.

## Plant model

For the mass–spring–damper system:

```text
m*x_ddot + c*x_dot + k*x = u(t)
```

with `m = 1 kg`, `c = 10 N·s/m`, `k = 20 N/m`, and target `1 m`.

At rest at 1 m, the spring force is `-k*x = -20 N`; therefore the actuator must apply
`+20 N` to hold the mass at the target.

## Experiments

Run commands from the repository root.

| Experiment | Purpose | Command |
|---|---|---|
| 01 Open loop | Observe the plant without feedback | `python -m experiments.01_open_loop` |
| 02 P-only | Sweep proportional gains and calculate P-only SSE | `python -m experiments.02_p_only` |
| 03 PI | Add integral action and study SSE/overshoot trade-off | `python -m experiments.03_pi_controller` |
| 04 PID | Combine P, I and D | `python -m experiments.04_pid_controller` |
| 05 PD | Hold Kp fixed and vary Kd; Ki = 0 | `python -m experiments.05_pd_controller` |
| 06 Saturation | Compare no saturation, saturation and anti-windup | `python -m experiments.06_actuator_saturation` |
| 07 Sensor noise | Add ±1 mm measurement noise | `python -m experiments.07_position_sensor` |
| 08 Derivative kick | Step the setpoint and observe derivative-on-error kick | `python -m experiments.08_derivative_kick` |
| 09 Slow update | Separate plant time step from controller update interval | `python -m experiments.09_slow_update_frequency` |
| 11 Ziegler–Nichols | Estimate ultimate gain/period and tune the third-order plant | `python -m experiments.11_ziegler_nichols` |

If a numeric-leading module filename cannot be imported in your Python environment, run the
script path from the repository root or rename it to a valid module name (for example
`exp_02_p_only.py`) and update the command.

## Setup

```bash
python3 -m venv PID_venv
source PID_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 matplotlib
```

If your `requirements.txt` pins compatible versions, use:

```bash
python -m pip install -r requirements.txt
```

Run the Below Command to Interact with UI based PID Concepts:

```bash
python app.py
```

Run from the project root so imports such as `from src...` resolve:

```bash
python -m experiments.02_p_only
```

## Interpreting results

- **Rise time:** shorter means faster initial approach, not necessarily better overall.
- **Overshoot:** lower is generally safer when crossing the target is costly.
- **Settling time:** shorter means the output enters and remains inside the selected tolerance band sooner.
- **Steady-state error:** lower means more accurate final tracking.
- **Control effort:** inspect peak and sustained force; a fast response that requires unrealistic force may not be deployable.

Select gains by considering all four assignment metrics and actuator effort, not one graph alone.

## Reproducibility checklist

- Keep plant parameters, initial conditions, gains, duration and `dt` fixed between comparable runs.
- Reset the controller before each independent run.
- Use a fixed random seed for sensor-noise comparisons.
- Keep actual position separate from noisy measured position.
- Record experiment settings with the generated figures.
- Re-run scripts before publishing; screenshots are useful evidence but do not replace reproducible outputs.

## Implementation notes

- The shared simulation API uses `position_sensor_obj`, not `sensor` or `use_position_sensor`.
- The standard simulation passes the same `dt` to the controller and plant. A slow-update
  experiment needs a separate controller period and must hold the previous force between updates.
- A ±30 N actuator limit clips the applied force. Record raw requested force separately.
- Conditional-integration anti-windup should pause integration only when the error pushes
  further into saturation; integration should resume when the error helps leave saturation.
- Initializing `previous_error` avoids only the first-sample startup spike. It does not remove
  a later setpoint-step derivative kick.

## Report

See `PID_Control_Internship_Report.pdf`. The report distinguishes supplied graph evidence
from analytical reference values and lists experiment graph sets that still need to be
generated from the local scripts.
