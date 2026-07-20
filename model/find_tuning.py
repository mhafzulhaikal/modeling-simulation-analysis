"""Find PID tuning parameters from open-loop step test data.

This script identifies a First-Order Plus Dead Time (FOPDT) model from
step response data using `model.fopdt` and calculates controller
tuning parameters based on classical tuning correlations from
`model/ctrlparams.py` (Corripio & Smith, 2006).

Usage:
    python -m model.find_tuning data.csv "Input Column" "Output Column"
"""

import argparse
import sys

import pandas as pd

from model.fopdt import FOPDTModel, identify_fopdt


def calculate_tunings(model: FOPDTModel) -> dict:
    """Calculate PID tuning parameters from an FOPDT model.

    Parameters
    ----------
    model : FOPDTModel
        FOPDT model parameters (K, tau, t0).

    Returns
    -------
    dict
        Nested dictionary of tuning parameters categorized by method.
    """
    K = model.K
    tau = max(model.tau, 1e-6)
    t0 = max(model.t0, 1e-6)

    if abs(K) < 1e-10:
        print("Warning: Process gain K is near zero. Cannot calculate tunings.")
        return {}

    ratio = t0 / tau
    tunings = {}

    # =========================================================================
    # 1. QUARTER DECAY RATIO
    # =========================================================================
    # PI
    Kc_PI_QDR = (0.9 / K) * (ratio) ** (-1)
    tauI_PI_QDR = 3.33 * t0

    # PID (series converted to ideal)
    Kc_prime = (1.2 / K) * (ratio) ** (-1)
    tauI_prime = 2.0 * t0
    tauD_prime = t0 / 2.0

    Kc_PID_QDR = Kc_prime * (1 + tauD_prime / tauI_prime)
    tauI_PID_QDR = tauI_prime + tauD_prime
    tauD_PID_QDR = tauD_prime * tauI_prime / (tauI_prime + tauD_prime)

    tunings["Quarter Decay Ratio (QDR)"] = {
        "P": {"Kc": (1.0 / K) * (ratio) ** (-1), "tauI": 0.0, "tauD": 0.0},
        "PI": {"Kc": Kc_PI_QDR, "tauI": tauI_PI_QDR, "tauD": 0.0},
        "PID": {"Kc": Kc_PID_QDR, "tauI": tauI_PID_QDR, "tauD": tauD_PID_QDR},
    }

    # =========================================================================
    # 2. MINIMUM IAE - DISTURBANCE
    # =========================================================================
    tunings["Minimum IAE (Disturbance)"] = {
        "P": {"Kc": (0.902 / K) * (ratio) ** (-0.985), "tauI": 0.0, "tauD": 0.0},
        "PI": {
            "Kc": (0.994 / K) * (ratio) ** (-0.986),
            "tauI": (tau / 0.608) * (ratio) ** 0.707,
            "tauD": 0.0,
        },
        "PID": {
            "Kc": (1.435 / K) * (ratio) ** (-0.921),
            "tauI": (tau / 0.878) * (ratio) ** 0.749,
            "tauD": 0.482 * tau * (ratio) ** 1.137,
        },
    }

    # =========================================================================
    # 3. MINIMUM IAE - SET POINT
    # =========================================================================
    # Guard against negative tauI for large t0/tau ratio
    denom_pi = 1.02 - 0.323 * ratio
    denom_pid = 0.740 - 0.130 * ratio

    tunings["Minimum IAE (Set Point)"] = {
        "PI": {
            "Kc": (0.758 / K) * (ratio) ** (-0.861),
            "tauI": tau / denom_pi if denom_pi > 0 else float("inf"),
            "tauD": 0.0,
        },
        "PID": {
            "Kc": (1.086 / K) * (ratio) ** (-0.869),
            "tauI": tau / denom_pid if denom_pid > 0 else float("inf"),
            "tauD": 0.348 * tau * (ratio) ** 0.914,
        },
    }

    # =========================================================================
    # 4. DAHLIN SYNTHESIS
    # =========================================================================
    tauC_PI = (2.0 / 3.0) * t0
    tauC_PID = (1.0 / 5.0) * t0

    # PI
    Kc_PI_Dahlin = tau / (K * tauC_PI)
    tauI_PI_Dahlin = tau

    # PID (series to ideal)
    Kc_prime_Dahlin = tau / (K * (t0 + tauC_PID))
    tauI_prime_Dahlin = tau
    tauD_prime_Dahlin = t0 / 2.0

    Kc_PID_Dahlin = Kc_prime_Dahlin * (1 + tauD_prime_Dahlin / tauI_prime_Dahlin)
    tauI_PID_Dahlin = tauI_prime_Dahlin + tauD_prime_Dahlin
    tauD_PID_Dahlin = (
        tauD_prime_Dahlin * tauI_prime_Dahlin / (tauI_prime_Dahlin + tauD_prime_Dahlin)
    )

    tunings["Dahlin Synthesis (Set Point)"] = {
        "PI": {"Kc": Kc_PI_Dahlin, "tauI": tauI_PI_Dahlin, "tauD": 0.0},
        "PID": {"Kc": Kc_PID_Dahlin, "tauI": tauI_PID_Dahlin, "tauD": tauD_PID_Dahlin},
    }

    # =========================================================================
    # 5. 5% OVERSHOOT
    # =========================================================================
    tunings["5% Overshoot"] = {
        "P": {"Kc": (0.5 / K) * (ratio) ** (-1), "tauI": 0.0, "tauD": 0.0}
    }

    return tunings


def print_tunings(tunings: dict):
    """Format and print the tunings to the console."""
    print("\n" + "=" * 80)
    print("  PID CONTROLLER TUNING PARAMETERS")
    print("=" * 80)

    if not tunings:
        print("  No tunings available.")
        return

    for method, controllers in tunings.items():
        print(f"\n--- {method} ---")
        for ctype, params in controllers.items():
            kc = params.get("Kc", 0.0)
            taui = params.get("tauI", 0.0)
            taud = params.get("tauD", 0.0)

            # Format handling for inf
            taui_str = f"{taui:10.4f}" if taui != float("inf") else "       inf"
            print(
                f"  {ctype:<4}: Kc = {kc:10.4f}, tauI = {taui_str}, tauD = {taud:10.4f}"
            )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Find PID tuning parameters from open-loop step test data."
    )
    parser.add_argument(
        "csv_file", help="Path to the CSV file containing step test data."
    )
    parser.add_argument(
        "input_col", help="Name of the input (manipulated variable) column."
    )
    parser.add_argument(
        "output_col", help="Name of the output (process variable) column."
    )
    parser.add_argument(
        "--time_col", default="Time", help="Name of the time column (default: Time)."
    )
    parser.add_argument(
        "--method",
        default="fast",
        choices=["two-point", "fast", "robust"],
        help="FOPDT identification method (default: fast).",
    )

    args = parser.parse_args()

    print(f"\nLoading data from {args.csv_file}...")
    try:
        df = pd.read_csv(args.csv_file)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    if args.time_col not in df.columns:
        print(
            f"Error: Time column '{args.time_col}' not found. Available columns: {df.columns.tolist()}"
        )
        sys.exit(1)
    if args.input_col not in df.columns:
        print(
            f"Error: Input column '{args.input_col}' not found. Available columns: {df.columns.tolist()}"
        )
        sys.exit(1)
    if args.output_col not in df.columns:
        print(
            f"Error: Output column '{args.output_col}' not found. Available columns: {df.columns.tolist()}"
        )
        sys.exit(1)

    time = df[args.time_col].to_numpy()
    input_data = df[args.input_col].to_numpy()
    output_data = df[args.output_col].to_numpy()

    print(f"\nIdentifying FOPDT model (method: {args.method})...")
    model, metrics, _ = identify_fopdt(
        time, input_data, output_data, method=args.method, verbose=True
    )

    tunings = calculate_tunings(model)
    print_tunings(tunings)


if __name__ == "__main__":
    main()
