from pathlib import Path
import traceback

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from src.pid_controller import PIDController
from src.simulation import run_simulation
from src.plants.mass_spring_damper import MassSpringDamper
from src.sensors.position_sensor import PositionSensor

# Optional black-box plant used by the Ziegler-Nichols dashboard experiment.
try:
    from src.plants.unknown_transfer_function import UnknownTransferFunction
except ImportError:
    UnknownTransferFunction = None


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


def _metric_summary(samples, setpoint):
    x = np.asarray([row["x"] for row in samples], dtype=float)
    t = np.asarray([row["t"] for row in samples], dtype=float)
    final_error = abs(setpoint - x[-1])
    i10 = np.flatnonzero(x >= 0.1 * setpoint)
    i90 = np.flatnonzero(x >= 0.9 * setpoint)
    rise = float(t[i90[0]] - t[i10[0]]) if len(i10) and len(i90) and i90[0] >= i10[0] else None
    overshoot = max(0.0, (float(np.max(x)) - setpoint) / max(abs(setpoint), 1e-9) * 100.0)
    band = 0.02 * max(abs(setpoint), 1e-9)
    outside = np.flatnonzero(np.abs(x - setpoint) > band)
    settling = float(t[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(t) else (0.0 if not len(outside) else None)
    return {
        "rise_time": rise,
        "overshoot_percent": overshoot,
        "settling_time": settling,
        "steady_state_error": final_error,
    }


def _make_mass_spring_plant(p):
    return MassSpringDamper(
        mass=p["mass"],
        damping=p["damping"],
        spring_constant=p["spring"],
        initial_position=0.0,
        initial_velocity=0.0,
        external_force=0.0,
    )


def _run_mass_spring(p):
    mode = p["experiment"]
    kp, ki, kd = p["kp"], p["ki"], p["kd"]
    if mode == "p":
        ki = kd = 0.0
    elif mode == "pi":
        kd = 0.0
    elif mode == "pd":
        ki = 0.0

    gains = {"kp": kp, "ki": ki, "kd": kd}
    limits = (-p["actuator_limit"], p["actuator_limit"])
    controller = PIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        setpoint=p["setpoint"],
        output_limits=limits,
        anti_windup=p["anti_windup"],
    )

    sensor = None
    if p["noise_mm"] > 0:
        sensor = PositionSensor(noise_limit=p["noise_mm"] / 1000.0, seed=42)

    results = run_simulation(
        plant=_make_mass_spring_plant(p),
        controller=controller,
        duration=p["duration"],
        dt=p["dt"],
        position_sensor_obj=sensor,
    )

    samples = []
    count = len(results["time"])
    for i in range(count):
        samples.append({
            "t": float(results["time"][i]),
            "x": float(results["position"][i]),
            "v": float(results["velocity"][i]),
            "meas": float(results.get("measured_position", results["position"])[i]),
            "e": float(results["error"][i]),
            "u": float(results["control_force"][i]),
            "raw": float(results.get("raw_control_force", results["control_force"])[i]),
            "p": float(results.get("proportional", np.zeros(count))[i]),
            "it": float(results.get("integral_term", np.zeros(count))[i]),
            "d": float(results.get("derivative_term", np.zeros(count))[i]),
        })

    return samples, gains, (
        "Computed by the project's Python modules: MassSpringDamper, "
        "PIDController, PositionSensor (when enabled), and run_simulation. "
        "Rise time is 10–90%; settling uses a ±2% band."
    )



def _run_special_mass_spring(p):
    """Run experiments that need a custom input/update schedule."""
    mode = p["experiment"]
    plant = _make_mass_spring_plant(p)
    kp, ki, kd = p["kp"], p["ki"], p["kd"]
    setpoint = p["setpoint"]
    dt = p["dt"]
    steps = int(round(p["duration"] / dt))
    limit = p["actuator_limit"]

    if mode == "open_loop":
        # A constant 20 N input demonstrates the open-loop plant response.
        force = 20.0
        controller = None
        note = "Open-loop experiment: a constant 20 N force is applied without feedback."
    else:
        if mode == "derivative_kick":
            ki, kd = p["ki"], p["kd"]
        controller = PIDController(
            kp=kp, ki=ki, kd=kd, setpoint=setpoint,
            output_limits=(-limit, limit),
            anti_windup=p["anti_windup"],
        )
        controller.reset()
        force = 0.0
        if mode == "derivative_kick":
            note = (
                "Derivative-kick experiment: the setpoint changes from 0.5 m "
                "to the selected target at t=1.0 s. Derivative is calculated "
                "from error, so the setpoint step creates a derivative spike."
            )
        else:
            controller_period = max(10 * dt, 0.01)
            note = (
                f"Slow-update experiment: plant physics uses dt={dt:g} s; "
                f"the PID controller updates every {controller_period:g} s "
                f"(approximately {1/controller_period:.1f} Hz). The previous "
                "force is held between controller updates."
            )

    rows = []
    previous_update_step = -1
    controller_period = max(10 * dt, 0.01)
    stride = max(1, int(round(controller_period / dt)))

    for step in range(steps + 1):
        t = step * dt
        actual = float(plant.position)

        if mode == "open_loop":
            applied = force
            p_term = i_term = d_term = 0.0
            raw = force
            error = setpoint - actual
            measured = actual
        else:
            if mode == "derivative_kick":
                controller.setpoint = 0.5 if t < 1.0 else setpoint
                should_update = True
            else:
                should_update = (step % stride == 0) or step == 0

            if should_update:
                measured = actual
                force = controller.calculate_output(
                    current_position=measured, dt=(dt if mode == "derivative_kick" else controller_period)
                )
                previous_update_step = step
            else:
                measured = actual

            applied = float(force)
            raw = float(getattr(controller, "raw_output", applied))
            p_term = float(getattr(controller, "proportional", 0.0))
            i_term = float(getattr(controller, "integral_term", 0.0))
            d_term = float(getattr(controller, "derivative_term", 0.0))
            error = float(getattr(controller, "error", setpoint - measured))

        rows.append({
            "t": float(t), "x": actual, "v": float(plant.velocity),
            "meas": float(measured), "e": float(error),
            "u": float(applied), "raw": float(raw),
            "p": float(p_term), "it": float(i_term), "d": float(d_term),
        })

        plant.external_force = applied
        if step < steps:
            plant.update_state(dt)

    gains = {"kp": float(kp), "ki": float(ki), "kd": float(kd)}
    return rows, gains, note

def _find_peaks(values):
    values = np.asarray(values)
    if len(values) < 3:
        return np.array([], dtype=int)
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1


def _run_zn(p):
    if UnknownTransferFunction is None:
        raise RuntimeError(
            "src/plants/unknown_transfer_function.py was not found. "
            "Copy the black-box plant file into src/plants/ first."
        )

    # ZN needs enough time to observe several oscillation cycles.
    # The dashboard's display duration can be short, so use a longer
    # identification run internally without changing the final display duration.
    identification_duration = max(float(p["duration"]), 30.0)

    sweep = np.arange(1.0, 12.01, 0.25)
    candidates = []

    for kp_test in sweep:
        plant = UnknownTransferFunction()
        controller = PIDController(
            kp=float(kp_test), ki=0.0, kd=0.0, setpoint=p["setpoint"]
        )
        results = run_simulation(
            plant, controller, identification_duration, p["dt"]
        )

        t = np.asarray(results["time"], dtype=float)
        y = np.asarray(results["position"], dtype=float)
        response = y - float(p["setpoint"])

        # Ignore the initial transient, then measure successive local maxima.
        start = int(0.25 * len(t))
        segment = response[start:]
        peaks = _find_peaks(segment)

        if len(peaks) < 4:
            continue

        peak_values = np.abs(segment[peaks])
        peak_times = t[start:][peaks]

        # Use the first and last groups of peaks to estimate whether the
        # oscillation envelope is decaying, growing, or approximately constant.
        group_size = max(2, len(peak_values) // 2)
        first_group = peak_values[:group_size]
        last_group = peak_values[-group_size:]

        first_amplitude = float(np.mean(first_group))
        last_amplitude = float(np.mean(last_group))

        if first_amplitude < 1e-8 or not np.isfinite(last_amplitude):
            continue

        amplitude_ratio = last_amplitude / first_amplitude
        score = abs(np.log(max(amplitude_ratio, 1e-12)))

        periods = np.diff(peak_times)
        tu = float(np.median(periods))

        # Reject clearly divergent or strongly decaying responses. Sustained
        # oscillation should have a peak-envelope ratio close to one.
        if not (0.70 <= amplitude_ratio <= 1.30):
            continue

        candidates.append((score, float(kp_test), tu, amplitude_ratio))

    if not candidates:
        raise RuntimeError(
            "Could not identify sustained oscillations. For this plant, "
            "try a smaller time step (e.g. 0.001 s) and ensure the black-box "
            "plant file is the third-order 1/(s+1)^3 model. The identification "
            "run already extends to at least 30 seconds."
        )

    score, ku, tu, amplitude_ratio = min(candidates, key=lambda item: item[0])

    # Classic closed-loop Ziegler–Nichols PID rules:
    # Kp = 0.6 Ku, Ti = Tu/2, Td = Tu/8.
    kp = 0.60 * ku
    ti = tu / 2.0
    td = tu / 8.0
    gains = {"kp": kp, "ki": kp / ti, "kd": kp * td}

    # Run the tuned controller for the user's requested display duration.
    plant = UnknownTransferFunction()
    controller = PIDController(
        kp=gains["kp"], ki=gains["ki"], kd=gains["kd"],
        setpoint=p["setpoint"],
    )
    results = run_simulation(plant, controller, p["duration"], p["dt"])

    samples = []
    n = len(results["time"])
    for i in range(n):
        samples.append({
            "t": float(results["time"][i]),
            "x": float(results["position"][i]),
            "v": float(results["velocity"][i]),
            "meas": float(results.get("measured_position", results["position"])[i]),
            "e": float(results["error"][i]),
            "u": float(results["control_force"][i]),
            "raw": float(results.get("raw_control_force", results["control_force"])[i]),
            "p": float(results.get("proportional", np.zeros(n))[i]),
            "it": float(results.get("integral_term", np.zeros(n))[i]),
            "d": float(results.get("derivative_term", np.zeros(n))[i]),
        })

    note = (
        f"Experimental black-box estimate: Ku≈{ku:.3f}, Tu≈{tu:.3f} s; "
        f"oscillation-envelope ratio={amplitude_ratio:.3f}. "
        f"Identification used {identification_duration:.1f} s to observe "
        f"multiple cycles. Classic ZN PID gains: Kp={kp:.4f}, "
        f"Ki={gains['ki']:.4f}, Kd={gains['kd']:.4f}. "
        "The final tuned response is plotted over the dashboard's selected "
        "duration. ZN gains are starting estimates, not guaranteed optimums."
    )
    return samples, gains, note


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.post("/api/simulate")
def simulate():
    try:
        raw = request.get_json(force=True)
        p = {
            "experiment": str(raw.get("experiment", "pid")),
            "kp": float(raw.get("kp", 50)),
            "ki": float(raw.get("ki", 20)),
            "kd": float(raw.get("kd", 5)),
            "setpoint": float(raw.get("setpoint", 1)),
            "duration": float(raw.get("duration", 8)),
            "dt": float(raw.get("dt", 0.002)),
            "mass": float(raw.get("mass", 1)),
            "damping": float(raw.get("damping", 10)),
            "spring": float(raw.get("spring", 20)),
            "noise_mm": float(raw.get("noise_mm", 0)),
            "actuator_limit": float(raw.get("actuator_limit", 100)),
            "anti_windup": bool(raw.get("anti_windup", True)),
        }
        if p["dt"] <= 0 or p["duration"] <= 0 or p["duration"] / p["dt"] > 500000:
            return jsonify(error="Choose positive duration/time step and at most 500,000 simulation steps."), 400
        if p["experiment"] == "zn":
            samples, gains, note = _run_zn(p)
        elif p["experiment"] in {"open_loop", "derivative_kick", "slow_update"}:
            samples, gains, note = _run_special_mass_spring(p)
        else:
            samples, gains, note = _run_mass_spring(p)
        return jsonify(
            samples=samples,
            gains=gains,
            metrics=_metric_summary(samples, p["setpoint"]),
            note=note,
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
