from pathlib import Path
import numpy as np

from src.pid_controller import PIDController
from src.simulation import run_simulation
from src.plants.unknown_transfer_function import UnknownTransferFunction
from src.performance import (
    calculate_performance,
    plot_position_response,
    plot_position_comparison,
    plot_control_force_comparison,
    plot_error_comparison,
)


# --------------------------- Experiment settings ---------------------------

SETPOINT = 1.0
DURATION = 35.0
DT = 0.001
KP_SWEEP = np.arange(2.0, 10.01, 0.25)

RESULTS_DIR = Path("results/10_ziegler_nichols_black_box")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_plant():
    return UnknownTransferFunction()


def make_p_controller(kp):
    return PIDController(
        kp=float(kp), ki=0.0, kd=0.0, setpoint=SETPOINT
    )


def find_local_maxima(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.array([], dtype=int)
    return np.where(
        (values[1:-1] > values[:-2]) &
        (values[1:-1] >= values[2:])
    )[0] + 1


def estimate_oscillation_quality(results, transient_fraction=0.50):
    """Return amplitude-change score and oscillation period from late peaks."""
    time = np.asarray(results["time"])
    position = np.asarray(results["position"])
    error_signal = position - SETPOINT

    start = int(len(time) * transient_fraction)
    late_error = error_signal[start:]
    late_time = time[start:]

    peak_idx = find_local_maxima(late_error)
    if len(peak_idx) < 4:
        return np.inf, np.nan

    amplitudes = late_error[peak_idx]
    peak_times = late_time[peak_idx]

    useful = amplitudes > 1e-5
    amplitudes = amplitudes[useful]
    peak_times = peak_times[useful]
    if len(amplitudes) < 4:
        return np.inf, np.nan

    midpoint = len(amplitudes) // 2
    first_mean = np.mean(amplitudes[:midpoint])
    second_mean = np.mean(amplitudes[midpoint:])
    if first_mean <= 1e-12:
        return np.inf, np.nan

    score = abs(second_mean - first_mean) / first_mean
    period = float(np.median(np.diff(peak_times)))
    return float(score), period


def calculate_zn_pid_gains(ku, tu):
    """Classic closed-loop ZN PID rules in parallel PID form."""
    kp = 0.60 * ku
    ti = tu / 2.0
    td = tu / 8.0
    ki = kp / ti
    kd = kp * td
    return {"kp": kp, "ki": ki, "kd": kd, "ti": ti, "td": td}


def run_p_gain_sweep():
    """Find the tested Kp whose late oscillation amplitudes are most constant."""
    records = []
    best = None

    print("\\nPHASE 1: P-only gain sweep")
    print("-" * 68)
    print(f"{'Kp':>8} {'Amplitude-change score':>25} {'Tu estimate (s)':>18}")

    for kp in KP_SWEEP:
        results = run_simulation(
            plant=make_plant(),
            controller=make_p_controller(kp),
            duration=DURATION,
            dt=DT,
        )
        score, tu = estimate_oscillation_quality(results)

        record = {"kp": float(kp), "score": score, "tu": tu, "results": results}
        records.append(record)
        print(f"{kp:8.2f} {score:25.6f} {tu:18.4f}")

        if np.isfinite(score) and np.isfinite(tu):
            if best is None or score < best["score"]:
                best = record

    if best is None:
        raise RuntimeError(
            "Could not detect enough late oscillation peaks. Check the shared "
            "simulation API and increase DURATION if necessary."
        )

    print(f"\\nEstimated Ku = {best['kp']:.4f}")
    print(f"Estimated Tu = {best['tu']:.4f} s")
    print(f"Amplitude-change score = {best['score']:.6f}")
    return best, records


def run_pid_case(name, gains):
    controller = PIDController(
        kp=gains["kp"],
        ki=gains["ki"],
        kd=gains["kd"],
        setpoint=SETPOINT,
    )
    results = run_simulation(
        plant=make_plant(),
        controller=controller,
        duration=DURATION,
        dt=DT,
    )
    return {
        "name": name,
        "results": results,
        "gains": gains,
        "metrics": calculate_performance(results, SETPOINT),
    }


def main():
    # PHASE 1: Black-box tuning. No transfer-function coefficients are used.
    best, sweep_records = run_p_gain_sweep()
    experimental_ku = best["kp"]
    experimental_tu = best["tu"]
    experimental_gains = calculate_zn_pid_gains(
        experimental_ku, experimental_tu
    )

    # PHASE 2: Reveal the model only for independent theoretical validation.
    # These validation values are not used by the black-box gain sweep.
    theoretical_ku = 8.0
    theoretical_tu = 2.0 * np.pi / np.sqrt(3.0)
    theoretical_gains = calculate_zn_pid_gains(
        theoretical_ku, theoretical_tu
    )

    print("\\nEXPERIMENTAL VS THEORETICAL ZN GAINS")
    print("-" * 58)
    print(f"{'Gain':<10}{'Experimental':>20}{'Theoretical':>20}")
    for key in ("kp", "ki", "kd"):
        print(
            f"{key.upper():<10}"
            f"{experimental_gains[key]:>20.6f}"
            f"{theoretical_gains[key]:>20.6f}"
        )

    experimental_case = run_pid_case("Experimental ZN", experimental_gains)
    theoretical_case = run_pid_case("Theoretical ZN", theoretical_gains)
    cases = [experimental_case, theoretical_case]

    print("\\nPERFORMANCE METRICS")
    print("-" * 58)
    for case in cases:
        print(f"\\n{case['name']}")
        print(
            f"Kp={case['gains']['kp']:.5f}, "
            f"Ki={case['gains']['ki']:.5f}, "
            f"Kd={case['gains']['kd']:.5f}"
        )
        for metric, value in case["metrics"].items():
            print(f"{metric}: {value}")

    # Existing plotting utilities from src/performance.py.
    plot_position_response(
        best["results"],
        setpoint=SETPOINT,
        output_dir=RESULTS_DIR,
        filename="ultimate_gain_oscillation.png",
    )

    comparison = {case["name"]: case["results"] for case in cases}

    plot_position_comparison(
        comparison,
        setpoint=SETPOINT,
        output_dir=RESULTS_DIR,
        filename="experimental_vs_theoretical_position.png",
    )
    plot_control_force_comparison(comparison, output_dir=RESULTS_DIR)
    plot_error_comparison(comparison, output_dir=RESULTS_DIR)

    print(f"\\nGraphs saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
