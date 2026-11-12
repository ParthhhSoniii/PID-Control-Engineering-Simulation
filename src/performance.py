"""Performance metrics and engineering plots for the PID simulation.

Run from the project root:
    python -m src.performance

The module reports settling time using both the ±2% and ±5% criteria.
The legacy "settling_time" metric is retained as an alias for ±2%.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------------------------

def _data(results):
    """Validate simulation results and return time and position arrays."""
    required = (
        "time", "position", "velocity",
        "acceleration", "error", "control_force"
    )
    missing = [key for key in required if key not in results]
    if missing:
        raise KeyError(f"Missing result fields: {missing}")

    time = np.asarray(results["time"], dtype=float)
    position = np.asarray(results["position"], dtype=float)

    if time.ndim != 1 or position.ndim != 1 or len(time) != len(position):
        raise ValueError("time and position must be 1D arrays of equal length.")

    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("At least two strictly increasing time samples are required.")

    return time, position


# ---------------------------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------------------------

def _first_crossing(time, values, threshold):
    """Return the interpolated time at which values first cross threshold."""
    indices = np.flatnonzero(values >= threshold)

    if len(indices) == 0:
        return float("nan")

    i = int(indices[0])

    if i == 0:
        return float(time[0])

    t0, t1 = time[i - 1], time[i]
    y0, y1 = values[i - 1], values[i]

    if np.isclose(y1, y0):
        return float(t1)

    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))


def calculate_rise_time(time, position, setpoint):
    """Return 10–90% rise time and the 10%/90% crossing timestamps."""
    time = np.asarray(time, dtype=float)
    position = np.asarray(position, dtype=float)

    initial = position[0]
    step = setpoint - initial

    if np.isclose(step, 0.0):
        return float("nan"), float("nan"), float("nan")

    progress = (position - initial) / step

    t10 = _first_crossing(time, progress, 0.10)
    t90 = _first_crossing(time, progress, 0.90)

    rise = (
        t90 - t10
        if np.isfinite(t10) and np.isfinite(t90)
        else float("nan")
    )

    return float(rise), float(t10), float(t90)


def calculate_overshoot(position, setpoint):
    """Return peak overshoot as a percentage of the commanded step."""
    position = np.asarray(position, dtype=float)
    step = setpoint - position[0]

    if np.isclose(step, 0.0):
        return 0.0

    progress = (position - position[0]) * np.sign(step)
    overshoot_amount = max(0.0, np.max(progress) - abs(step))

    return float(100.0 * overshoot_amount / abs(step))


def calculate_settling_time(time, position, setpoint, tolerance=0.02):
    """Return first time after which response stays within the tolerance band.

    The tolerance is expressed as a fraction of the commanded step:
    0.02 means ±2%, and 0.05 means ±5%.

    Returns NaN if the response does not enter and remain inside the band
    before the simulation ends.
    """
    time = np.asarray(time, dtype=float)
    position = np.asarray(position, dtype=float)

    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1.")

    step_size = abs(setpoint - position[0])

    if np.isclose(step_size, 0.0):
        return float(time[0])

    band = tolerance * step_size
    outside = np.flatnonzero(np.abs(position - setpoint) > band)

    if len(outside) == 0:
        return float(time[0])

    last_outside = int(outside[-1])

    if last_outside == len(time) - 1:
        return float("nan")

    return float(time[last_outside + 1])


def calculate_steady_state_error(position, setpoint, tail_fraction=0.05):
    """Return setpoint minus mean position over the final fraction of samples."""
    position = np.asarray(position, dtype=float)

    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be greater than 0 and at most 1.")

    count = max(1, int(np.ceil(len(position) * tail_fraction)))
    final_position = float(np.mean(position[-count:]))

    return float(setpoint - final_position)


def calculate_performance(results, setpoint):
    """Calculate rise time, overshoot, both settling times, and SSE."""
    time, position = _data(results)

    rise, t10, t90 = calculate_rise_time(time, position, setpoint)

    settling_2 = calculate_settling_time(
        time, position, setpoint, tolerance=0.02
    )
    settling_5 = calculate_settling_time(
        time, position, setpoint, tolerance=0.05
    )

    return {
        "rise_time": rise,
        "overshoot_percent": calculate_overshoot(position, setpoint),
        "settling_time_2_percent": settling_2,
        "settling_time_5_percent": settling_5,
        # Backward-compatible alias: settling_time means ±2%.
        "settling_time": settling_2,
        "steady_state_error": calculate_steady_state_error(position, setpoint),
        "t10": t10,
        "t90": t90,
    }


# ---------------------------------------------------------------------
# PLOT SAVING
# ---------------------------------------------------------------------

def _finish(fig, output_dir, filename):
    """Display a figure or save it in the requested output directory."""
    if output_dir is None:
        plt.show()
    else:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------
# POSITION RESPONSE PLOT
# ---------------------------------------------------------------------

def plot_position_response(
    results,
    setpoint,
    output_dir=None,
    filename="position_response.png",
):
    """Plot response with 10–90% rise time, overshoot, SSE, and both bands."""
    time, position = _data(results)
    metrics = calculate_performance(results, setpoint)

    initial_position = position[0]
    step_size = setpoint - initial_position
    direction = np.sign(step_size)

    if np.isclose(step_size, 0.0):
        raise ValueError("Setpoint must differ from the initial position.")

    # Rise-time levels and timestamps.
    y10 = initial_position + 0.10 * step_size
    y90 = initial_position + 0.90 * step_size
    t10 = metrics["t10"]
    t90 = metrics["t90"]
    rise_time = metrics["rise_time"]

    overshoot = metrics["overshoot_percent"]
    settling_time_2 = metrics["settling_time_2_percent"]
    settling_time_5 = metrics["settling_time_5_percent"]
    steady_state_error = metrics["steady_state_error"]

    # Mean of the final 5% of samples is used as the estimated
    # steady-state position, matching the SSE calculation.
    tail_count = max(1, int(np.ceil(len(position) * 0.05)))
    steady_state_position = float(np.mean(position[-tail_count:]))

    # Peak in the direction of the commanded step.
    peak_index = int(np.argmax(
        (position - initial_position) * direction
    ))
    peak_time = float(time[peak_index])
    peak_position = float(position[peak_index])

    # Both tolerance bands are based on the commanded step size.
    band_2 = abs(step_size) * 0.02
    band_5 = abs(step_size) * 0.05

    lower_2, upper_2 = setpoint - band_2, setpoint + band_2
    lower_5, upper_5 = setpoint - band_5, setpoint + band_5

    fig, ax = plt.subplots(figsize=(15, 8))

    # Actual response and target.
    ax.plot(
        time, position,
        linewidth=2.5,
        label="Actual Position",
        zorder=3,
    )
    ax.axhline(
        setpoint,
        linestyle="--",
        linewidth=1.8,
        label=f"Target Position ({setpoint:.2f} m)",
        zorder=2,
    )

    # Draw the wider 5% band first, then the narrower 2% band.
    ax.axhspan(
        lower_5, upper_5,
        alpha=0.12,
        label=f"±5% Settling Band ({lower_5:.3f}–{upper_5:.3f} m)",
        zorder=0,
    )
    ax.axhspan(
        lower_2, upper_2,
        alpha=0.25,
        label=f"±2% Settling Band ({lower_2:.3f}–{upper_2:.3f} m)",
        zorder=1,
    )

    # 10–90% rise-time levels.
    ax.axhline(y10, linestyle=":", alpha=0.65, linewidth=1.2)
    ax.axhline(y90, linestyle=":", alpha=0.65, linewidth=1.2)

    if np.isfinite(t10) and np.isfinite(t90):
        ax.axvline(t10, linestyle=":", alpha=0.65, linewidth=1.2)
        ax.axvline(t90, linestyle=":", alpha=0.65, linewidth=1.2)

        ax.scatter(
            [t10, t90], [y10, y90],
            s=75,
            zorder=5,
            label="10% and 90% Rise-Time Points",
        )

        # Horizontal double arrow marks the interval from t10 to t90.
        arrow_y = initial_position + 0.035 * step_size
        ax.annotate(
            "",
            xy=(t90, arrow_y),
            xytext=(t10, arrow_y),
            arrowprops=dict(arrowstyle="<->", linewidth=1.8),
        )
        ax.text(
            (t10 + t90) / 2,
            arrow_y - 0.045 * abs(step_size),
            f"Rise Time (10–90%) = {rise_time:.4f} s",
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.2),
        )
    else:
        ax.text(
            0.03, 0.80,
            "Rise Time (10–90%): NOT ACHIEVED",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.2),
        )

    # Overshoot: vertical double arrow precisely from setpoint to peak.
    ax.scatter(
        peak_time, peak_position,
        s=120,
        zorder=6,
        label="Peak Position",
    )
    if direction * (peak_position - setpoint) > 0:
        ax.annotate(
            "",
            xy=(peak_time, peak_position),
            xytext=(peak_time, setpoint),
            arrowprops=dict(arrowstyle="<->", linewidth=2),
            zorder=7,
        )
        ax.annotate(
            f"Overshoot = {overshoot:.2f}%",
            xy=(peak_time, (peak_position + setpoint) / 2),
            xytext=(peak_time + 0.35, peak_position),
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.2),
            fontsize=11,
        )

    # Steady-state position and SSE arrow.
    ax.axhline(
        steady_state_position,
        linestyle=":",
        linewidth=1.8,
        label=f"Steady-State Position ({steady_state_position:.4f} m)",
    )

    # Place the SSE arrow near the right edge, but inside the axes.
    x_range = time[-1] - time[0]
    sse_x = time[0] + 0.94 * x_range

    ax.annotate(
        "",
        xy=(sse_x, setpoint),
        xytext=(sse_x, steady_state_position),
        arrowprops=dict(arrowstyle="<->", linewidth=2),
        zorder=7,
    )
    ax.annotate(
        f"Steady-State Error = {abs(steady_state_error):.4f} m",
        xy=(sse_x, (setpoint + steady_state_position) / 2),
        xytext=(
            time[0] + 0.68 * x_range,
            (setpoint + steady_state_position) / 2 - 0.22 * abs(step_size),
        ),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.2),
        fontsize=11,
    )

    # Report both settling-time criteria.
    settling_2_text = (
        f"±2% Settling Time: {settling_time_2:.4f} s"
        if np.isfinite(settling_time_2)
        else "±2% Settling Time: NOT ACHIEVED"
    )
    settling_5_text = (
        f"±5% Settling Time: {settling_time_5:.4f} s"
        if np.isfinite(settling_time_5)
        else "±5% Settling Time: NOT ACHIEVED"
    )

    ax.text(
        0.98, 0.96,
        settling_2_text + "\n" + settling_5_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.2),
    )

    # Labels, grid, and legend.
    ax.set_title(
        "PID Controller Performance Evaluation",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_ylabel("Position (metres)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    _finish(fig, output_dir, filename)

    return metrics


# ---------------------------------------------------------------------
# OTHER ENGINEERING PLOTS
# ---------------------------------------------------------------------

def _plot_signal(results, key, title, ylabel, filename, output_dir):
    """Plot one recorded simulation signal against time."""
    time, _ = _data(results)
    values = np.asarray(results[key], dtype=float)

    if len(values) != len(time):
        raise ValueError(f"{key} must have the same length as time.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, values, linewidth=1.8)
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _finish(fig, output_dir, filename)


def plot_velocity_response(results, output_dir=None):
    _plot_signal(
        results, "velocity", "Velocity Response", "Velocity (m/s)",
        "velocity_response.png", output_dir,
    )


def plot_acceleration_response(results, output_dir=None):
    _plot_signal(
        results, "acceleration", "Acceleration Response",
        "Acceleration (m/s²)", "acceleration_response.png", output_dir,
    )


def plot_control_force(results, output_dir=None):
    _plot_signal(
        results, "control_force", "PID Control Force",
        "Control force (N)", "control_force.png", output_dir,
    )


def plot_error_response(results, output_dir=None):
    _plot_signal(
        results, "error", "Tracking Error", "Position error (m)",
        "error_response.png", output_dir,
    )


def plot_pid_components(results, output_dir=None):
    """Plot P/I/D terms; requires simulation.py to record these fields."""
    keys = ("proportional", "integral_term", "derivative_term")
    missing = [key for key in keys if key not in results]

    if missing:
        raise KeyError(
            f"PID component fields not recorded by simulation.py: {missing}"
        )

    time = np.asarray(results["time"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = {
        "proportional": "P contribution",
        "integral_term": "I contribution",
        "derivative_term": "D contribution",
    }

    for key in keys:
        values = np.asarray(results[key], dtype=float)

        if len(values) != len(time):
            raise ValueError(f"{key} must have the same length as time.")

        ax.plot(time, values, label=labels[key])

    ax.set(
        title="PID Component Contributions",
        xlabel="Time (s)",
        ylabel="Force contribution (N)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _finish(fig, output_dir, "pid_components.png")


def plot_phase_portrait(results, output_dir=None):
    """Plot velocity against position."""
    time, position = _data(results)
    velocity = np.asarray(results["velocity"], dtype=float)

    if len(velocity) != len(time):
        raise ValueError("velocity must have the same length as time.")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(position, velocity, linewidth=1.8)
    ax.scatter(position[0], velocity[0], label="Start")
    ax.scatter(position[-1], velocity[-1], marker="x", s=80, label="End")
    ax.set(
        title="Phase Portrait",
        xlabel="Position (m)",
        ylabel="Velocity (m/s)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _finish(fig, output_dir, "phase_portrait.png")


def plot_controller_comparison(experiments, setpoint, output_dir=None):
    """Plot position responses and return metrics for each controller."""
    if not experiments:
        raise ValueError("Provide at least one experiment result.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    all_metrics = {}

    for name, results in experiments.items():
        time, position = _data(results)
        ax.plot(time, position, linewidth=1.8, label=name)
        all_metrics[name] = calculate_performance(results, setpoint)

    ax.axhline(setpoint, linestyle="--", color="black", label="Setpoint")
    ax.set(
        title="Controller Response Comparison",
        xlabel="Time (s)",
        ylabel="Position (m)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _finish(fig, output_dir, "controller_comparison.png")

    return all_metrics

def plot_pd_comparisons(experiments, setpoint, output_dir=None):
    """Create position, velocity, acceleration, force, error and phase overlays."""
    if not experiments:
        raise ValueError("Provide at least one experiment result.")

    metrics = plot_controller_comparison(experiments, setpoint, output_dir)

    if output_dir is not None:
        folder = Path(output_dir)
        old_graph = folder / "controller_comparison.png"
        new_graph = folder / "position_comparison.png"
        if old_graph.exists():
            old_graph.replace(new_graph)

    plot_velocity_comparison(experiments, output_dir)
    plot_acceleration_comparison(experiments, output_dir)
    plot_control_force_comparison(experiments, output_dir)
    plot_error_comparison(experiments, output_dir)
    plot_phase_portrait_comparison(experiments, output_dir)
    return metrics



# ---------------------------------------------------------------------
# COMPLETE PERFORMANCE REPORT
# ---------------------------------------------------------------------

def generate_performance_report(results, setpoint, output_dir="results/graphs"):
    """Save standard plots and return all performance metrics."""
    metrics = plot_position_response(results, setpoint, output_dir)

    plot_velocity_response(results, output_dir)
    plot_acceleration_response(results, output_dir)
    plot_control_force(results, output_dir)
    plot_error_response(results, output_dir)
    plot_phase_portrait(results, output_dir)

    # Include the P/I/D component plot if simulation.py records these arrays.
    if all(
        key in results
        for key in ("proportional", "integral_term", "derivative_term")
    ):
        plot_pid_components(results, output_dir)

    return metrics


# ---------------------------------------------------------------------
# RUN AS A MODULE
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Run from the project root using: python -m src.performance
    from plants.mass_spring_damper import MassSpringDamper
    from pid_controller import PIDController
    from simulation import run_simulation

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

    simulation_results = run_simulation(
        plant,
        controller,
        duration=5.0,
        dt=0.001,
    )

    metrics = generate_performance_report(
        simulation_results,
        setpoint=controller.setpoint,
    )

    print("Performance metrics:")
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            if np.isfinite(value):
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: NOT ACHIEVED")

# ---------------------------------------------------------------------
# MULTI-EXPERIMENT COMPARISON PLOTS
# ---------------------------------------------------------------------

def _validate_experiments(experiments):
    """Validate {legend_label: simulation_results} input."""
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("experiments must be a non-empty {label: results} dictionary.")
    normalized = {}
    for label, results in experiments.items():
        _data(results)
        normalized[str(label)] = results
    return normalized


def plot_position_comparison(
    experiments, setpoint, output_dir=None,
    filename="position_comparison.png",
):
    """Overlay position responses; legend includes overshoot and SSE."""
    experiments = _validate_experiments(experiments)
    fig, ax = plt.subplots(figsize=(15, 8))
    all_metrics = {}

    for label, results in experiments.items():
        time, position = _data(results)
        metrics = calculate_performance(results, setpoint)
        all_metrics[label] = metrics
        ax.plot(
            time, position, linewidth=2.0,
            label=(
                f"{label} | OS={metrics['overshoot_percent']:.2f}%"
                f" | SSE={metrics['steady_state_error']:.4f} m"
            ),
        )

    _, first_position = _data(next(iter(experiments.values())))
    step_size = setpoint - first_position[0]
    if np.isclose(step_size, 0.0):
        raise ValueError("Setpoint must differ from the initial position.")

    band_2 = abs(step_size) * 0.02
    band_5 = abs(step_size) * 0.05
    ax.axhline(setpoint, linestyle="--", color="black", linewidth=1.8,
               label=f"Target ({setpoint:.2f} m)")
    ax.axhspan(setpoint-band_5, setpoint+band_5, color="gray", alpha=0.10,
               label="±5% settling band")
    ax.axhspan(setpoint-band_2, setpoint+band_2, color="gray", alpha=0.20,
               label="±2% settling band")
    ax.set(title="Position Response Comparison",
           xlabel="Time (s)", ylabel="Position (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    _finish(fig, output_dir, filename)
    return all_metrics


def _plot_signal_comparison(
    experiments, key, title, ylabel, filename, output_dir=None
):
    """Shared implementation for multi-run time-series comparisons."""
    experiments = _validate_experiments(experiments)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, results in experiments.items():
        time, _ = _data(results)
        values = np.asarray(results[key], dtype=float)
        if values.ndim != 1 or len(values) != len(time):
            raise ValueError(f"{key} for {label!r} must match the time array.")
        ax.plot(time, values, linewidth=1.8, label=label)
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    _finish(fig, output_dir, filename)


def plot_velocity_comparison(
    experiments, output_dir=None, filename="velocity_comparison.png"
):
    _plot_signal_comparison(
        experiments, "velocity", "Velocity Response Comparison",
        "Velocity (m/s)", filename, output_dir
    )


def plot_acceleration_comparison(
    experiments, output_dir=None, filename="acceleration_comparison.png"
):
    _plot_signal_comparison(
        experiments, "acceleration", "Acceleration Response Comparison",
        "Acceleration (m/s²)", filename, output_dir
    )


def plot_control_force_comparison(
    experiments, output_dir=None, filename="control_force_comparison.png"
):
    _plot_signal_comparison(
        experiments, "control_force", "Control Force Comparison",
        "Control force (N)", filename, output_dir
    )


def plot_error_comparison(
    experiments, output_dir=None, filename="error_comparison.png"
):
    _plot_signal_comparison(
        experiments, "error", "Tracking Error Comparison",
        "Position error (m)", filename, output_dir
    )


def plot_phase_portrait_comparison(
    experiments, output_dir=None, filename="phase_portrait_comparison.png"
):
    """Overlay velocity-versus-position trajectories."""
    experiments = _validate_experiments(experiments)
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, results in experiments.items():
        _, position = _data(results)
        velocity = np.asarray(results["velocity"], dtype=float)
        if velocity.ndim != 1 or len(velocity) != len(position):
            raise ValueError(f"velocity for {label!r} must match position.")
        ax.plot(position, velocity, linewidth=1.8, label=label)
    ax.set(title="Phase Portrait Comparison",
           xlabel="Position (m)", ylabel="Velocity (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    _finish(fig, output_dir, filename)


def plot_pid_components_comparison(
    experiments, output_dir=None, filename="pid_components_comparison.png"
):
    """Overlay P, I and D contributions for multiple configurations."""
    experiments = _validate_experiments(experiments)
    keys = (
        ("proportional", "P contribution"),
        ("integral_term", "I contribution"),
        ("derivative_term", "D contribution"),
    )
    missing = {
        label: [key for key, _ in keys if key not in results]
        for label, results in experiments.items()
    }
    missing = {label: fields for label, fields in missing.items() if fields}
    if missing:
        raise KeyError(f"PID component fields are missing: {missing}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (key, component_label) in zip(axes, keys):
        for label, results in experiments.items():
            time = np.asarray(results["time"], dtype=float)
            values = np.asarray(results[key], dtype=float)
            if values.ndim != 1 or len(values) != len(time):
                raise ValueError(f"{key} for {label!r} must match time.")
            ax.plot(time, values, linewidth=1.6, label=label)
        ax.set_ylabel(f"{component_label} (N)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_title("PID Component Comparison")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _finish(fig, output_dir, filename)


def generate_comparison_report(
    experiments, setpoint, output_dir="results/graphs/comparison"
):
    """Generate all multi-configuration plots and return per-run metrics.

    Pass a dictionary mapping descriptive labels (e.g. 'Kp=10') to
    result dictionaries returned by the existing run_simulation() engine.
    All existing single-run functions and signatures remain unchanged.
    """
    experiments = _validate_experiments(experiments)
    metrics = plot_position_comparison(
        experiments, setpoint, output_dir=output_dir
    )
    plot_velocity_comparison(experiments, output_dir=output_dir)
    plot_acceleration_comparison(experiments, output_dir=output_dir)
    plot_control_force_comparison(experiments, output_dir=output_dir)
    plot_error_comparison(experiments, output_dir=output_dir)
    plot_phase_portrait_comparison(experiments, output_dir=output_dir)

    component_keys = ("proportional", "integral_term", "derivative_term")
    if all(
        all(key in results for key in component_keys)
        for results in experiments.values()
    ):
        plot_pid_components_comparison(experiments, output_dir=output_dir)

    return metrics
