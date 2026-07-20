"""Calculate performance metrics from step-test data and validate Python against HYSYS.

Provides functions to compute dynamic response performance metrics and robust
statistical validation metrics comparing Python simulation results to Aspen HYSYS
process simulator reference data.

Usage:
    python analysis/calculate_performance.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import datetime
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from model import StepInfo

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "data", "step_test_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

THRESH_FIC = 0.02       # ±2 %
THRESH_LIC = 0.02       # ±2 %  (setpoint)
THRESH_LIC_DIST = 0.01  # ±1 %  (disturbance)
THRESH_TIC = 0.02       # ±2 %  (setpoint)
THRESH_TIC_DIST = 0.01  # ±1 %  (disturbance)

FIC_STEP_TIME = 600.0
LIC_SP_STEP_TIME = 600.0
LIC_DIST_STEP_TIME = 3600.0
LIC_DIST_END_TIME = 25200.0
TIC_DIST_STEP_TIME = 600.0
TIC_DIST_END_TIME = 20000.0
TIC_SP_STEP_TIME = 20600.0

FIC_SP_INITIAL = 50.0
FIC_SP_FINAL = 60.0
LIC_SP_INITIAL = 50.0
LIC_SP_FINAL = 60.0
TIC_SP_INITIAL = 50.0
TIC_SP_FINAL = 60.0

# Variable mappings for Python and HYSYS comparison
VALIDATION_VARIABLES = ["PV"]

COLUMN_MAPPINGS = {
    "FIC-100": {
        "PV": {"py": "FT-100 - C", "hy": "FIC-100 - PV"},
        "SP": {"py": "FSP-100 - R", "hy": "FIC-100 - SP"},
        "MV": {"py": "FC-100 - M", "hy": "FIC-100 - OP"}
    },
    "FIC-101": {
        "PV": {"py": "FT-101 - C", "hy": "FIC-101 - PV"},
        "SP": {"py": "FSP-101 - R", "hy": "FIC-101 - SP"},
        "MV": {"py": "FC-101 - M", "hy": "FIC-101 - OP"}
    },
    "FIC-102": {
        "PV": {"py": "FT-102 - C", "hy": "FIC-102 - PV"},
        "SP": {"py": "FSP-102 - R", "hy": "FIC-102 - SP"},
        "MV": {"py": "FC-102 - M", "hy": "FIC-102 - OP"}
    },
    "LIC-100": {
        "PV": {"py": "LT-100 - C", "hy": "LIC-100 - PV"},
        "SP": {"py": "LSP-100 - R", "hy": "LIC-100 - SP"},
        "MV": {"py": "LC-100 - M", "hy": "LIC-100 - OP"}
    },
    "TIC-100": {
        "PV": {"py": "TT-100 - C", "hy": "TIC-100 - PV"},
        "SP": {"py": "TSP-100 - R", "hy": "TIC-100 - SP"},
        "MV": {"py": "TCV-100 - VP", "hy": "TIC-100 - OP"}
    }
}

VALIDATION_CASES = [
    {
        "loop": "FIC-100",
        "scenario": "Setpoint",
        "py_path": os.path.join(DATA_DIR, "FIC-100", "Python", "FIC_100_Setpoint.CSV"),
        "hy_path": os.path.join(DATA_DIR, "FIC-100", "HYSYS", "FIC_100_setpoint.CSV")
    },
    {
        "loop": "FIC-101",
        "scenario": "Setpoint",
        "py_path": os.path.join(DATA_DIR, "FIC-101", "Python", "FIC_101_Setpoint.CSV"),
        "hy_path": os.path.join(DATA_DIR, "FIC-101", "HYSYS", "FIC_101_setpoint.CSV")
    },
    {
        "loop": "FIC-102",
        "scenario": "Setpoint",
        "py_path": os.path.join(DATA_DIR, "FIC-102", "Python", "FIC_102_Setpoint.CSV"),
        "hy_path": os.path.join(DATA_DIR, "FIC-102", "HYSYS", "FIC_102_setpoint.CSV")
    },
    {
        "loop": "LIC-100",
        "scenario": "Setpoint-Tight",
        "py_path": os.path.join(DATA_DIR, "LIC-100", "Python", "LIC_100_Tight_Setpoint.CSV"),
        "hy_path": os.path.join(DATA_DIR, "LIC-100", "HYSYS", "LIC_100_Tight_setpoint.CSV")
    },
    {
        "loop": "LIC-100",
        "scenario": "Setpoint-Averaging",
        "py_path": os.path.join(DATA_DIR, "LIC-100", "Python", "LIC_100_Averaging_Setpoint.CSV"),
        "hy_path": os.path.join(DATA_DIR, "LIC-100", "HYSYS", "LIC_100_Averaging_setpoint.CSV")
    },
    {
        "loop": "LIC-100",
        "scenario": "Disturbance-Tight",
        "py_path": os.path.join(DATA_DIR, "LIC-100", "Python", "LIC_100_Tight_Disturbance_Scenario.CSV"),
        "hy_path": os.path.join(DATA_DIR, "LIC-100", "HYSYS", "LIC_100_Tight_disturbance_scenario.CSV")
    },
    {
        "loop": "LIC-100",
        "scenario": "Disturbance-Averaging",
        "py_path": os.path.join(DATA_DIR, "LIC-100", "Python", "LIC_100_Averaging_Disturbance_Scenario.CSV"),
        "hy_path": os.path.join(DATA_DIR, "LIC-100", "HYSYS", "LIC_100_Averaging_disturbance_scenario.CSV")
    },
    {
        "loop": "TIC-100",
        "scenario": "Setpoint-Syn",
        "py_path": os.path.join(DATA_DIR, "TIC-100", "Python", "setpoint-tuning", "TIC_100_Syn.CSV"),
        "hy_path": os.path.join(DATA_DIR, "TIC-100", "HYSYS", "setpoint-tuning", "TIC_100_Syn.CSV")
    },
    {
        "loop": "TIC-100",
        "scenario": "Setpoint-QDR",
        "py_path": os.path.join(DATA_DIR, "TIC-100", "Python", "setpoint-tuning", "TIC_100_QDR.CSV"),
        "hy_path": os.path.join(DATA_DIR, "TIC-100", "HYSYS", "setpoint-tuning", "TIC_100_QDR.CSV")
    },
    {
        "loop": "TIC-100",
        "scenario": "Setpoint-IAE",
        "py_path": os.path.join(DATA_DIR, "TIC-100", "Python", "setpoint-tuning", "TIC_100_IAE.CSV"),
        "hy_path": os.path.join(DATA_DIR, "TIC-100", "HYSYS", "setpoint-tuning", "TIC_100_IAE.CSV")
    },
    {
        "loop": "TIC-100",
        "scenario": "Disturbance-Syn",
        "py_path": os.path.join(DATA_DIR, "TIC-100", "Python", "disturbance-tuning", "TIC_100_Syn.CSV"),
        "hy_path": os.path.join(DATA_DIR, "TIC-100", "HYSYS", "disturbance-tuning", "TIC_100_Syn.CSV")
    },
    {
        "loop": "TIC-100",
        "scenario": "Disturbance-QDR",
        "py_path": os.path.join(DATA_DIR, "TIC-100", "Python", "disturbance-tuning", "TIC_100_QDR.CSV"),
        "hy_path": os.path.join(DATA_DIR, "TIC-100", "HYSYS", "disturbance-tuning", "TIC_100_QDR.CSV")
    },
    {
        "loop": "TIC-100",
        "scenario": "Disturbance-IAE",
        "py_path": os.path.join(DATA_DIR, "TIC-100", "Python", "disturbance-tuning", "TIC_100_IAE.CSV"),
        "hy_path": os.path.join(DATA_DIR, "TIC-100", "HYSYS", "disturbance-tuning", "TIC_100_IAE.CSV")
    }
]

# =============================================================================
# Helper Functions
# =============================================================================

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file, strip whitespace from columns, and raise clear errors if missing.
    
    Parameters:
        path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Stripped columns DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required simulation output file is missing: '{path}'.\n"
            f"Please verify that simulation data has been generated."
        )
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{path}'. File might be corrupted. Error: {e}")
        
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    return df


def _load(path: str) -> pd.DataFrame:
    """Compatibility wrapper around load_csv."""
    return load_csv(path)


def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract column as floating point numpy array, raising key errors if missing.
    
    Parameters:
        df: pd.DataFrame
        col: str, column name
        
    Returns:
        np.ndarray: float array
    """
    if col not in df.columns:
        raise KeyError(
            f"Required column '{col}' is missing in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )
    return np.asarray(df[col].values, dtype=float)


def _si(time, y, u, y0, yf, step_time, end_time=None, thresh=0.01):
    """Create StepInfo performance metrics analysis container."""
    return StepInfo(
        time=time,
        y=y,
        u=u,
        y_initial=y0,
        y_final=yf,
        step_time=step_time,
        end_time=end_time,
        SettlingTimeThreshold=thresh,
    )


def align_time_series(df_python: pd.DataFrame, df_hysys: pd.DataFrame, time_col: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Align Python results onto the HYSYS time grid using linear interpolation.
    
    Parameters:
        df_python: Python simulation DataFrame.
        df_hysys: HYSYS simulation DataFrame (reference grid).
        time_col: Column name representing the time coordinates.
        
    Returns:
        t_ref: np.ndarray, HYSYS reference time coordinates.
        df_py_aligned: pd.DataFrame, Python DataFrame interpolated onto t_ref.
        df_hysys: pd.DataFrame, HYSYS reference DataFrame.
    """
    if time_col not in df_python.columns:
        raise KeyError(f"Time column '{time_col}' not found in Python CSV.")
    if time_col not in df_hysys.columns:
        raise KeyError(f"Time column '{time_col}' not found in HYSYS CSV.")
    
    t_ref = np.asarray(df_hysys[time_col].values, dtype=float)
    t_py = np.asarray(df_python[time_col].values, dtype=float)
    
    df_py_aligned = pd.DataFrame({time_col: t_ref})
    for col in df_python.columns:
        if col != time_col:
            fp = np.asarray(df_python[col].values, dtype=float)
            df_py_aligned[col] = np.interp(t_ref, t_py, fp)
            
    return t_ref, df_py_aligned, df_hysys


def calculate_validation_metrics(y_hysys: np.ndarray, y_python: np.ndarray, epsilon: float = 1e-8) -> Dict[str, Any]:
    """Calculate detailed mathematical validation metrics between HYSYS (ref) and Python.
    
    Checks edge cases, division-by-zero, and issues specific warnings.
    
    Parameters:
        y_hysys: np.ndarray, HYSYS process values (reference).
        y_python: np.ndarray, Python simulation values.
        epsilon: float, threshold for avoiding divisions by zero.
        
    Returns:
        dict: calculated validation metrics and warning list.
    """
    mask = ~np.isnan(y_hysys) & ~np.isnan(y_python)
    yh = y_hysys[mask]
    yp = y_python[mask]
    n_total = len(yh)
    
    if n_total == 0:
        return {
            "N": 0, "MeanHYSYS": float("nan"), "MeanPython": float("nan"),
            "MAE": float("nan"), "RMSE": float("nan"), "MaxAbsErr": float("nan"),
            "NRMSE_range": float("nan"), "MAPE": float("nan"), "ExcludedMAPE": 0,
            "SMAPE": float("nan"), "ExcludedSMAPE": 0, "MBE": float("nan"),
            "PBIAS": float("nan"), "R2": float("nan"), "Warnings": ["No valid overlapping data samples."]
        }
        
    warnings = []
    
    mean_hy = np.mean(yh)
    mean_py = np.mean(yp)
    mae = np.mean(np.abs(yp - yh))
    rmse = np.sqrt(np.mean((yp - yh)**2))
    max_abs_err = np.max(np.abs(yp - yh))
    
    # NRMSE (range-normalized)
    hy_range = np.max(yh) - np.min(yh)
    if hy_range <= epsilon:
        nrmse = float("nan")
        warnings.append("HYSYS reference range is zero; NRMSE range set to NaN.")
    else:
        nrmse = (rmse / hy_range) * 100.0
        
    # MAPE (exclude reference values close to 0)
    mape_mask = np.abs(yh) > epsilon
    n_excluded_mape = int(np.sum(~mape_mask))
    if np.any(mape_mask):
        mape = np.mean(np.abs((yh[mape_mask] - yp[mape_mask]) / yh[mape_mask])) * 100.0
    else:
        mape = float("nan")
        warnings.append("All HYSYS reference values are near zero; MAPE set to NaN.")
        
    # SMAPE (exclude points where both are near zero)
    denom = np.abs(yh) + np.abs(yp)
    smape_mask = denom > epsilon
    n_excluded_smape = int(np.sum(~smape_mask))
    if np.any(smape_mask):
        smape = np.mean(2.0 * np.abs(yp[smape_mask] - yh[smape_mask]) / denom[smape_mask]) * 100.0
    else:
        smape = float("nan")
        warnings.append("All denominator elements in SMAPE are near zero; SMAPE set to NaN.")
        
    # MBE (Mean Bias Error)
    mbe = np.mean(yp - yh)
    
    # PBIAS
    sum_yh = np.sum(yh)
    if np.abs(sum_yh) <= epsilon:
        pbias = float("nan")
        warnings.append("Sum of HYSYS reference values is near zero; PBIAS set to NaN.")
    else:
        pbias = 100.0 * np.sum(yp - yh) / sum_yh
        
    # R2
    ss_res = np.sum((yh - yp)**2)
    ss_tot = np.sum((yh - mean_hy)**2)
    if ss_tot <= epsilon:
        r2 = float("nan")
        warnings.append("HYSYS data variance is zero; R² set to NaN.")
    else:
        r2 = 1.0 - (ss_res / ss_tot)
        
    return {
        "N": n_total,
        "MeanHYSYS": mean_hy,
        "MeanPython": mean_py,
        "MAE": mae,
        "RMSE": rmse,
        "MaxAbsErr": max_abs_err,
        "NRMSE_range": nrmse,
        "MAPE": mape,
        "ExcludedMAPE": n_excluded_mape,
        "SMAPE": smape,
        "ExcludedSMAPE": n_excluded_smape,
        "MBE": mbe,
        "PBIAS": pbias,
        "R2": r2,
        "Warnings": warnings
    }


def calculate_integral_error(time: np.ndarray, error: np.ndarray) -> Dict[str, float]:
    """Calculate integral error performance indices (IAE, ISE, ITAE) using numerical integration.
    
    Parameters:
        time: np.ndarray, time vector in seconds.
        error: np.ndarray, error signal (e.g., SP - PV).
        
    Returns:
        dict: containing 'IAE', 'ISE', and 'ITAE' values.
    """
    if len(time) < 2:
        return {"IAE": float("nan"), "ISE": float("nan"), "ITAE": float("nan")}
    
    dt = np.diff(time)
    abs_err = np.abs(error)
    sq_err = error ** 2
    
    # Trapezoidal integration intervals
    iae_intervals = 0.5 * (abs_err[:-1] + abs_err[1:]) * dt
    ise_intervals = 0.5 * (sq_err[:-1] + sq_err[1:]) * dt
    
    t_mid = 0.5 * (time[:-1] + time[1:])
    itae_intervals = t_mid * iae_intervals
    
    return {
        "IAE": float(np.sum(iae_intervals)),
        "ISE": float(np.sum(ise_intervals)),
        "ITAE": float(np.sum(itae_intervals))
    }




def write_summary(validation_results: List[Dict[str, Any]], txt_path: str) -> None:
    """Write human-readable validation summary report including a scientific interpretation.
    
    Parameters:
        validation_results: List of validation metrics dicts.
        txt_path: Path to validation_summary.txt.
    """
    W_LEN = 90
    W_SEP = "=" * W_LEN
    
    lines = []
    lines.append(W_SEP)
    lines.append("  MATHEMATICAL VALIDATION SUMMARY  -  PYTHON VS HYSYS (PV ONLY)")
    lines.append("  Biodiesel Reactor Control System Validation")
    lines.append(W_SEP)
    lines.append("")
    
    hdr = (
        f"  {'Loop':<7} {'Scenario':<21} {'Var':<3}"
        f" {'N':>6} {'MAE':>6} {'RMSE':>6} {'MAPE%':>7} {'SMAPE%':>7} {'PBIAS%':>7} {'R2':>6}"
    )
    sep = (
        "  " + "-" * 7 + " " + "-" * 21 + " " + "-" * 3 +
        " " + "-" * 6 + " " + "-" * 6 + " " + "-" * 6 + " " + "-" * 7 + " " + "-" * 7 + " " + "-" * 7 + " " + "-" * 6
    )
    lines.append(hdr)
    lines.append(sep)
    
    for res in validation_results:
        loop = res["loop"]
        scenario = res["scenario"]
        var = res["variable"]
        n = res["N"]
        
        def _fmt(v):
            return "---" if np.isnan(v) or np.isinf(v) else f"{v:.3f}"
            
        mae = _fmt(res["MAE"])
        rmse = _fmt(res["RMSE"])
        mape = _fmt(res["MAPE"])
        smape = _fmt(res["SMAPE"])
        pbias = _fmt(res["PBIAS"])
        r2 = _fmt(res["R2"])
        
        lines.append(
            f"  {loop:<7} {scenario:<21} {var:<3}"
            f" {n:>6} {mae:>6} {rmse:>6} {mape:>7} {smape:>7} {pbias:>7} {r2:>6}"
        )
    lines.append(sep)
    lines.append("")
    
    # Scientific Interpretation Section
    lines.append(W_SEP)
    lines.append("  SCIENTIFIC INTERPRETATION GUIDE")
    lines.append(W_SEP)
    lines.append("  This section provides a guide for interpreting the validation metrics and plots:")
    lines.append("")
    lines.append("  1. MEAN ABSOLUTE PERCENTAGE ERROR (MAPE):")
    lines.append("     - MAPE is useful as a relative error metric, but it should not be used alone.")
    lines.append("       Specifically, when HYSYS values are close to zero, MAPE can inflate")
    lines.append("       disproportionately or become undefined due to division by zero.")
    lines.append("")
    lines.append("  2. MEAN ABSOLUTE ERROR (MAE) & ROOT MEAN SQUARE ERROR (RMSE):")
    lines.append("     - MAE and RMSE show error magnitude in the original unit.")
    lines.append("     - MAE represents the average linear error, whereas RMSE squashes deviations")
    lines.append("       quadratically, making it more sensitive to transient spikes and large lags.")
    lines.append("")

    lines.append("  4. MEAN BIAS ERROR (MBE) & PERCENT BIAS (PBIAS):")
    lines.append("     - Bias or PBIAS indicates whether Python systematically overpredicts or underpredicts HYSYS.")
    lines.append("       Positive bias shows overprediction; negative bias indicates underprediction.")
    lines.append("")
    lines.append("  5. COEFFICIENT OF DETERMINATION (R2):")
    lines.append("     - R2 indicates whether Python follows the variation pattern of HYSYS, but")
    lines.append("       it should not be interpreted as agreement by itself. Perfect correlation")
    lines.append("       (R2=1) can still occur alongside structural offsets (biases).")
    lines.append("")

    lines.append("")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Validation summary saved ->  {txt_path}")


def _write_summary(registry: dict, txt_path: str, csv_path: str) -> None:
    """Original StepInfo summary writer, preserved for backward compatibility."""
    W80 = "=" * 80
    W80d = "-" * 80
    W80h = "#" * 80
    NA = "---"

    def _fv(v, fmt):
        return (
            NA if (np.isnan(float(v)) or np.isinf(float(v))) else format(float(v), fmt)
        )

    lines = [
        W80,
        "  STEP TEST DATA  -  PERFORMANCE METRICS ANALYSIS",
        "  Biodiesel Reactor Control System",
        W80,
        f"  Generated   :  {datetime.date.today()}",
        f"  Data root   :  {DATA_DIR}/",
        "  Loops       :  FIC-100, FIC-101, FIC-102, LIC-100, TIC-100",
        f"  Scenarios   :  {len(registry)}",
        W80,
    ]

    current_loop = None

    for (loop, source, scenario), si in registry.items():
        if loop != current_loop:
            current_loop = loop
            lines += [
                "",
                W80h,
                f"  {loop}",
                W80h,
            ]

        step_str = f"step_time = {si.step_time:.1f} s"
        end_str = f"  end_time = {si.end_time:.1f} s" if si.end_time else ""
        rng_str = f"{si.y_initial:.4g} -> {si.yfinal:.4g} %TO"
        lines += [
            "",
            f"  [ {source} | {scenario} | {step_str}{end_str} | {rng_str} ]",
            W80d,
        ]
        for line in repr(si).split("\n"):
            lines.append("  " + line)

    # --- Comparison table ---
    lines += [
        "",
        W80h,
        "  COMPARISON TABLE  -  KEY METRICS",
        W80h,
        "  Settling band: Setpoint 2.0% (FIC/LIC/TIC)  |  Disturbance 1.0%",
        "",
    ]

    hdr = (
        f"  {'Loop':<7}  {'Source':<6}  {'Scenario':<27}"
        f"  {'tR [s]':>9}  {'tS [s]':>9}  {'OS [%]':>8}"
        f"  {'IAE':>10}"
    )
    sep = (
        "  "
        + "-" * 7
        + "  "
        + "-" * 6
        + "  "
        + "-" * 27
        + "  "
        + "-" * 9
        + "  "
        + "-" * 9
        + "  "
        + "-" * 8
        + "  "
        + "-" * 10
    )
    lines += [hdr, sep]

    prev_loop = None
    for (loop, source, scenario), si in registry.items():
        if prev_loop is not None and loop != prev_loop:
            lines.append(sep)
        prev_loop = loop
        d = si.to_dict()
        rt = _fv(d["RiseTime"], ".2f")
        ts = _fv(d["SettlingTime"], ".2f")
        os_ = _fv(d["Overshoot"], ".4f")
        iae = _fv(d["IAE"], ".2f")
        lines.append(
            f"  {loop:<7}  {source:<6}  {scenario:<27}"
            f"  {rt:>9}  {ts:>9}  {os_:>8}"
            f"  {iae:>10}"
        )

    lines += [sep]

    # === DEVIATION ANALYSIS: Python vs HYSYS ===
    metrics_labels = ["tR [s]", "tS [s]", "OS [%]", "IAE"]

    py_entries: dict[tuple, StepInfo] = {}
    hy_entries: dict[tuple, StepInfo] = {}
    for (loop, source, scenario), si in registry.items():
        key = (loop, scenario)
        if source == "Python":
            py_entries[key] = si
        elif source == "HYSYS":
            hy_entries[key] = si

    matched_keys = sorted(
        set(py_entries.keys()) & set(hy_entries.keys()),
        key=lambda k: list(registry.keys()).index((k[0], "Python", k[1])),
    )

    if matched_keys:
        lines += [
            "",
            W80h,
            "  DEVIATION ANALYSIS  -  PYTHON vs HYSYS (SCENARIO MAPE)",
            W80h,
            "  Reference: HYSYS   |   Scenario MAPE = |Python - HYSYS| / |HYSYS| x 100",
            "",
        ]

        dev_hdr = (
            f"  {'Loop':<7}  {'Scenario':<27}"
            f"  {'tR [%]':>9}  {'tS [%]':>9}  {'OS [%]':>9}"
            f"  {'IAE [%]':>9}  {'MAPE [%]':>9}"
        )
        dev_sep = (
            "  "
            + "-" * 7
            + "  "
            + "-" * 27
            + "  "
            + "-" * 9
            + "  "
            + "-" * 9
            + "  "
            + "-" * 9
            + "  "
            + "-" * 9
            + "  "
            + "-" * 9
        )
        lines += [dev_hdr, dev_sep]
        
        # Phase 1: Compute all raw deviations for all scenario pairs
        raw_devs = {}
        raw_errs = {}
        for key in matched_keys:
            loop, scenario = key
            si_py = py_entries[key]
            si_hy = hy_entries[key]
            d_py = si_py.to_dict()
            d_hy = si_hy.to_dict()

            raw_devs[key] = {}
            raw_errs[key] = {}
            for ml, py_v, hy_v in [
                ("tR [s]", d_py["RiseTime"], d_hy["RiseTime"]),
                ("tS [s]", d_py["SettlingTime"], d_hy["SettlingTime"]),
                ("OS [%]", d_py["Overshoot"], d_hy["Overshoot"]),
                ("IAE", d_py["IAE"], d_hy["IAE"]),
            ]:
                try:
                    pv = float(py_v)
                    hv = float(hy_v)
                    if np.isnan(pv) or np.isnan(hv) or np.isinf(pv) or np.isinf(hv):
                        raw_devs[key][ml] = float("nan")
                        raw_errs[key][ml] = float("nan")
                    else:
                        err = pv - hv
                        raw_errs[key][ml] = err
                        if abs(hv) < 1e-12:
                            raw_devs[key][ml] = 0.0 if abs(pv) < 1e-12 else float("nan")
                        else:
                            raw_devs[key][ml] = abs(err) / abs(hv) * 100.0
                except (TypeError, ValueError):
                    raw_devs[key][ml] = float("nan")
                    raw_errs[key][ml] = float("nan")

        # Phase 2: Identify scenario keys for each group and find metrics valid for ALL scenarios in that group
        loop_group_keys = {"FIC": [], "LIC": [], "TIC": []}
        for key in matched_keys:
            loop, _ = key
            if loop in ("FIC-100", "FIC-101", "FIC-102"):
                loop_group_keys["FIC"].append(key)
            elif loop == "LIC-100":
                loop_group_keys["LIC"].append(key)
            elif loop == "TIC-100":
                loop_group_keys["TIC"].append(key)

        group_valid_metrics = {}
        for lg, keys in loop_group_keys.items():
            valid_metrics = []
            for ml in metrics_labels:
                # Metric is valid only if present (not NaN) for ALL scenario pairs in the group
                if keys and all(not np.isnan(raw_devs[k][ml]) for k in keys):
                    valid_metrics.append(ml)
            group_valid_metrics[lg] = valid_metrics

        # Phase 3: Populating Scenario deviations and overall structures
        loop_group_devs: dict[str, dict[str, list[float]]] = {
            "FIC": {k: [] for k in metrics_labels},
            "LIC": {k: [] for k in metrics_labels},
            "TIC": {k: [] for k in metrics_labels},
        }

        overall_errors = {
            k: {"abs_pct": [], "abs": [], "sq": []} for k in metrics_labels
        }

        for key in matched_keys:
            loop, scenario = key
            if loop in ("FIC-100", "FIC-101", "FIC-102"):
                loop_group = "FIC"
            elif loop == "LIC-100":
                loop_group = "LIC"
            elif loop == "TIC-100":
                loop_group = "TIC"
            else:
                loop_group = loop

            valid_metrics = group_valid_metrics.get(loop_group, [])

            parts = []
            for ml in metrics_labels:
                pct = raw_devs[key][ml]
                err = raw_errs[key][ml]
                
                # Exclude if not fully present in all scenarios of this group
                if ml not in valid_metrics or np.isnan(pct):
                    parts.append(f"  {NA:>9}")
                else:
                    parts.append(f"  {pct:>9.2f}")
                    loop_group_devs[loop_group][ml].append(pct)
                    overall_errors[ml]["abs_pct"].append(pct)
                    overall_errors[ml]["abs"].append(abs(err))
                    overall_errors[ml]["sq"].append(err**2)

            # Scenario MAPE (average of ONLY valid metrics in the scenario)
            scenario_valid_pcts = [raw_devs[key][ml] for ml in valid_metrics if not np.isnan(raw_devs[key][ml])]
            if scenario_valid_pcts:
                row_mape = np.mean(scenario_valid_pcts)
                parts.append(f"  {row_mape:>9.2f}")
            else:
                parts.append(f"  {NA:>9}")

            lines.append(f"  {loop:<7}  {scenario:<27}" + "".join(parts))

        lines.append(dev_sep)

        # --- Per-loop-group average absolute deviation (MAPE) ---
        lines += [
            "",
            "  MEAN ABSOLUTE PERCENTAGE ERROR (MAPE) BY LOOP GROUP",
            dev_sep,
        ]
        lines.append(
            f"  {'Loop':<7}  {'# Pairs':<27}"
            f"  {'tR [%]':>9}  {'tS [%]':>9}  {'OS [%]':>9}"
            f"  {'IAE [%]':>9}  {'MAPE [%]':>9}"
        )
        lines.append(dev_sep)

        loop_group_mapes = {}
        loop_group_single_mape = {}
        loop_group_order = ["FIC", "LIC", "TIC"]

        for loop_group in loop_group_order:
            lg_devs = loop_group_devs.get(loop_group, {})
            loop_group_mapes[loop_group] = {}
            n_pairs = max(len(v) for v in lg_devs.values()) if lg_devs else 0
            parts = []

            # 1. Compute column-wise MAPEs for the group (only for valid metrics)
            for ml in metrics_labels:
                vals = lg_devs.get(ml, [])
                clean_vals = [v for v in vals if not np.isnan(v)]
                if ml in group_valid_metrics.get(loop_group, []) and clean_vals:
                    mean_abs = np.mean(clean_vals)
                    loop_group_mapes[loop_group][ml] = mean_abs
                    parts.append(f"  {mean_abs:>9.2f}")
                else:
                    loop_group_mapes[loop_group][ml] = float("nan")
                    parts.append(f"  {NA:>9}")

            # 2. Compute loop MAPE (average across valid metrics for this loop group)
            valid_metric_mapes = [loop_group_mapes[loop_group][ml] for ml in group_valid_metrics.get(loop_group, []) if not np.isnan(loop_group_mapes[loop_group][ml])]
            if valid_metric_mapes:
                lg_mape = np.mean(valid_metric_mapes)
                loop_group_single_mape[loop_group] = lg_mape
                parts.append(f"  {lg_mape:>9.2f}")
            else:
                loop_group_single_mape[loop_group] = float("nan")
                parts.append(f"  {NA:>9}")

            lines.append(
                f"  {loop_group:<7}  {str(n_pairs) + ' pairs':<27}" + "".join(parts)
            )

        lines.append(dev_sep)

        # --- Overall mean absolute deviation (MAPE) ---
        parts = []
        overall_mape_by_metric = {}
        for ml in metrics_labels:
            # Average only over loop groups where this metric was valid/fully-present
            group_vals = [loop_group_mapes[lg][ml] for lg in loop_group_order if ml in group_valid_metrics.get(lg, []) and not np.isnan(loop_group_mapes[lg][ml])]
            if group_vals:
                avg_mape = np.mean(group_vals)
                overall_mape_by_metric[ml] = avg_mape
                parts.append(f"  {avg_mape:>9.2f}")
            else:
                overall_mape_by_metric[ml] = float("nan")
                parts.append(f"  {NA:>9}")

        # Overall loop MAPE across all 3 loop groups (FIC, LIC, TIC)
        valid_group_mapes = [loop_group_single_mape[lg] for lg in loop_group_order if not np.isnan(loop_group_single_mape[lg])]
        if valid_group_mapes:
            grand_overall_mape = np.mean(valid_group_mapes)
            parts.append(f"  {grand_overall_mape:>9.2f}")
        else:
            grand_overall_mape = float("nan")
            parts.append(f"  {NA:>9}")

        lines.append(
            f"  {'OVERALL':<7}  {str(len(matched_keys)) + ' pairs':<27}"
            + "".join(parts)
        )
        lines.append(dev_sep)

        # --- AGGREGATE ERROR METRICS (ALL PAIRS) ---
        agg_sep = "  " + "-" * 12 + "  " + "-" * 12 + "  " + "-" * 14 + "  " + "-" * 14
        lines += [
            "",
            "  OVERALL ERROR METRICS (ALL PAIRS)",
            agg_sep,
            f"  {'Metric':<12}  {'MAPE [%]':>12}  {'MAE (Abs)':>14}  {'RMSE (Abs)':>14}",
            agg_sep,
        ]

        for ml in metrics_labels:
            errs = overall_errors[ml]
            if errs["abs_pct"]:
                mape = overall_mape_by_metric[ml]
                mae = np.mean(errs["abs"])
                rmse = np.sqrt(np.mean(errs["sq"]))
                lines.append(f"  {ml:<12}  {mape:>12.2f}  {mae:>14.2f}  {rmse:>14.2f}")
            else:
                lines.append(f"  {ml:<12}  {NA:>12}  {NA:>14}  {NA:>14}")

        lines.append(agg_sep)

        # --- Grand overall single deviation metric ---
        # Calculated from the loop group MAPEs
        if not np.isnan(grand_overall_mape):
            lines += [
                "",
                f"  Grand Mean Absolute Percentage Error (MAPE, all metrics):  "
                f"{grand_overall_mape:.2f} %",
            ]

    lines += ["", W80, "  END OF REPORT", W80, ""]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved   ->  {txt_path}")

    # CSV export
    rows = []
    for (loop, source, scenario), si in registry.items():
        row = {"Loop": loop, "Source": source, "Scenario": scenario}
        row.update(si.to_dict())
        nm = si.normalized_metrics()
        row["PerformanceIndex"] = (
            nm.get("PerformanceIndex", float("nan")) if nm else float("nan")
        )
        rows.append(row)
    pd.DataFrame(rows).set_index(["Loop", "Source", "Scenario"]).to_csv(csv_path)
    print(f"CSV saved      ->  {csv_path}")


def calculate_all_performance() -> Dict[str, Any]:
    """Execute performance metrics calculations for all control loops using StepInfo.
    
    Returns:
        dict: registry and StepInfo instances of all scenarios.
    """
    registry = {}

    # ----------------------------------------------------
    # FIC-100
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  FIC-100  —  Setpoint Tracking")
    print("=" * 62)

    df_fic100_py = _load(os.path.join(DATA_DIR, "FIC-100", "Python", "FIC_100_Setpoint.CSV"))
    t = _arr(df_fic100_py, "Time")
    C = _arr(df_fic100_py, "FT-100 - C")
    R = _arr(df_fic100_py, "FSP-100 - R")
    si_fic100_py = _si(t, C, R, FIC_SP_INITIAL, FIC_SP_FINAL, FIC_STEP_TIME, thresh=THRESH_FIC)
    print("\n[Python]")
    print(si_fic100_py)
    registry[("FIC-100", "Python", "Setpoint")] = si_fic100_py

    df_fic100_hy = _load(os.path.join(DATA_DIR, "FIC-100", "HYSYS", "FIC_100_setpoint.CSV"))
    t = _arr(df_fic100_hy, "Time")
    C = _arr(df_fic100_hy, "FIC-100 - PV")
    R = _arr(df_fic100_hy, "FIC-100 - SP")
    si_fic100_hy = _si(t, C, R, FIC_SP_INITIAL, FIC_SP_FINAL, FIC_STEP_TIME, thresh=THRESH_FIC)
    print("\n[HYSYS]")
    print(si_fic100_hy)
    registry[("FIC-100", "HYSYS", "Setpoint")] = si_fic100_hy

    # ----------------------------------------------------
    # FIC-101
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  FIC-101  —  Setpoint Tracking")
    print("=" * 62)

    df_fic101_py = _load(os.path.join(DATA_DIR, "FIC-101", "Python", "FIC_101_Setpoint.CSV"))
    t, C, R = (_arr(df_fic101_py, c) for c in ("Time", "FT-101 - C", "FSP-101 - R"))
    si_fic101_py = _si(t, C, R, FIC_SP_INITIAL, FIC_SP_FINAL, FIC_STEP_TIME, thresh=THRESH_FIC)
    print("\n[Python]")
    print(si_fic101_py)
    registry[("FIC-101", "Python", "Setpoint")] = si_fic101_py

    df_fic101_hy = _load(os.path.join(DATA_DIR, "FIC-101", "HYSYS", "FIC_101_setpoint.CSV"))
    t, C, R = (_arr(df_fic101_hy, c) for c in ("Time", "FIC-101 - PV", "FIC-101 - SP"))
    si_fic101_hy = _si(t, C, R, FIC_SP_INITIAL, FIC_SP_FINAL, FIC_STEP_TIME, thresh=THRESH_FIC)
    print("\n[HYSYS]")
    print(si_fic101_hy)
    registry[("FIC-101", "HYSYS", "Setpoint")] = si_fic101_hy

    # ----------------------------------------------------
    # FIC-102
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  FIC-102  —  Setpoint Tracking")
    print("=" * 62)

    df_fic102_py = _load(os.path.join(DATA_DIR, "FIC-102", "Python", "FIC_102_Setpoint.CSV"))
    t, C, R = (_arr(df_fic102_py, c) for c in ("Time", "FT-102 - C", "FSP-102 - R"))
    si_fic102_py = _si(t, C, R, FIC_SP_INITIAL, FIC_SP_FINAL, FIC_STEP_TIME, thresh=THRESH_FIC)
    print("\n[Python]")
    print(si_fic102_py)
    registry[("FIC-102", "Python", "Setpoint")] = si_fic102_py

    df_fic102_hy = _load(os.path.join(DATA_DIR, "FIC-102", "HYSYS", "FIC_102_setpoint.CSV"))
    t, C, R = (_arr(df_fic102_hy, c) for c in ("Time", "FIC-102 - PV", "FIC-102 - SP"))
    si_fic102_hy = _si(t, C, R, FIC_SP_INITIAL, FIC_SP_FINAL, FIC_STEP_TIME, thresh=THRESH_FIC)
    print("\n[HYSYS]")
    print(si_fic102_hy)
    registry[("FIC-102", "HYSYS", "Setpoint")] = si_fic102_hy

    # ----------------------------------------------------
    # LIC-100 Setpoint
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  LIC-100  —  Setpoint Tracking  (Tight & Averaging)")
    print("=" * 62)

    lic_sp_si = {}
    for label, fname in (
        ("Tight", "LIC_100_Tight_Setpoint.CSV"),
        ("Averaging", "LIC_100_Averaging_Setpoint.CSV"),
    ):
        df = _load(os.path.join(DATA_DIR, "LIC-100", "Python", fname))
        t, C, R = (_arr(df, c) for c in ("Time", "LT-100 - C", "LSP-100 - R"))
        si = _si(t, C, R, LIC_SP_INITIAL, LIC_SP_FINAL, LIC_SP_STEP_TIME, thresh=THRESH_LIC)
        lic_sp_si[label] = si
        print(f"\n[Python  {label}]")
        print(si)
        registry[("LIC-100", "Python", f"Setpoint-{label}")] = si

    lic_sp_hy_si = {}
    for label, fname in (
        ("Tight", "LIC_100_Tight_setpoint.CSV"),
        ("Averaging", "LIC_100_Averaging_setpoint.CSV"),
    ):
        df = _load(os.path.join(DATA_DIR, "LIC-100", "HYSYS", fname))
        t, C, R = (_arr(df, c) for c in ("Time", "LIC-100 - PV", "LIC-100 - SP"))
        si = _si(t, C, R, LIC_SP_INITIAL, LIC_SP_FINAL, LIC_SP_STEP_TIME, thresh=THRESH_LIC)
        lic_sp_hy_si[label] = si
        print(f"\n[HYSYS  {label}]")
        print(si)
        registry[("LIC-100", "HYSYS", f"Setpoint-{label}")] = si

    lic_dist_si = {}
    lic_dist_hy_si = {}

    # ----------------------------------------------------
    # TIC-100 Setpoint (Python)
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  TIC-100  —  Setpoint Tracking  (Python)")
    print("=" * 62)

    tic_sp_py_si = {}
    si_nested = {}
    for tuning in ("Syn", "QDR", "IAE"):
        df = _load(os.path.join(DATA_DIR, "TIC-100", "Python", "setpoint-tuning", f"TIC_100_{tuning}.CSV"))
        t, C, R = (_arr(df, c) for c in ("Time", "TT-100 - C", "TSP-100 - R"))
        si = _si(t, C, R, TIC_SP_INITIAL, TIC_SP_FINAL, TIC_SP_STEP_TIME, thresh=THRESH_TIC)
        tic_sp_py_si[tuning] = si
        print(f"\n[Python  {tuning}]")
        print(si)
        registry[("TIC-100", "Python", f"Setpoint-{tuning}")] = si

        si_dist = _si(t, C, 50.0, y0=50.0, yf=50.0, step_time=TIC_DIST_STEP_TIME, end_time=TIC_DIST_END_TIME, thresh=THRESH_TIC_DIST)
        si_nested[tuning] = {"Disturbance": si_dist, "Setpoint": si}

    # ----------------------------------------------------
    # TIC-100 Disturbance (Python)
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  TIC-100  —  Disturbance Rejection  (Python)")
    print("=" * 62)

    tic_dist_py_si = {}
    tic_dist_py_sp_si = {}
    for tuning in ("Syn", "QDR", "IAE"):
        df = _load(os.path.join(DATA_DIR, "TIC-100", "Python", "disturbance-tuning", f"TIC_100_{tuning}.CSV"))
        t, C, R = (_arr(df, c) for c in ("Time", "TT-100 - C", "TSP-100 - R"))
        si = _si(t, C, 50.0, y0=50.0, yf=50.0, step_time=TIC_DIST_STEP_TIME, end_time=TIC_DIST_END_TIME, thresh=THRESH_TIC_DIST)
        tic_dist_py_si[tuning] = si
        tic_dist_py_sp_si[tuning] = _si(t, C, R, TIC_SP_INITIAL, TIC_SP_FINAL, TIC_SP_STEP_TIME, thresh=THRESH_TIC)
        print(f"\n[Python  {tuning}]  (window {TIC_DIST_STEP_TIME:.0f}–{TIC_DIST_END_TIME:.0f} s)")
        print(si)
        registry[("TIC-100", "Python", f"Disturbance-{tuning}")] = si

    # ----------------------------------------------------
    # TIC-100 Dual-Step Python Performance Index logging
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  TIC-100  —  Dual-Step  (Python)")
    print("=" * 62)
    for tuning in ("Syn", "QDR", "IAE"):
        si_d = si_nested[tuning]["Disturbance"]
        si_s = si_nested[tuning]["Setpoint"]
        pi_d = si_d.normalized_metrics().get("PerformanceIndex", float("nan"))
        pi_s = si_s.normalized_metrics().get("PerformanceIndex", float("nan"))
        print(f"  {tuning:<4}  Disturbance PI = {pi_d:6.2f}   |   Setpoint PI = {pi_s:6.2f}")

    # ----------------------------------------------------
    # TIC-100 HYSYS (Setpoint + Disturbance)
    # ----------------------------------------------------
    print("\n" + "=" * 62)
    print("  TIC-100  —  HYSYS  (Setpoint + Disturbance)")
    print("=" * 62)

    tic_sp_hy_si = {}
    tic_dist_hy_si = {}
    tic_dist_hy_sp_si = {}
    si_nested_hy = {}
    for tuning in ("Syn", "QDR", "IAE"):
        # Setpoint
        df = _load(os.path.join(DATA_DIR, "TIC-100", "HYSYS", "setpoint-tuning", f"TIC_100_{tuning}.CSV"))
        t, C, R = (_arr(df, c) for c in ("Time", "TIC-100 - PV", "TIC-100 - SP"))
        si_sp = _si(t, C, R, TIC_SP_INITIAL, TIC_SP_FINAL, TIC_SP_STEP_TIME, thresh=THRESH_TIC)
        tic_sp_hy_si[tuning] = si_sp
        print(f"\n[HYSYS  {tuning}  setpoint]")
        print(si_sp)
        registry[("TIC-100", "HYSYS", f"Setpoint-{tuning}")] = si_sp

        si_dist_hy_dual = _si(t, C, 50.0, y0=50.0, yf=50.0, step_time=TIC_DIST_STEP_TIME, end_time=TIC_DIST_END_TIME, thresh=THRESH_TIC_DIST)
        si_nested_hy[tuning] = {"Disturbance": si_dist_hy_dual, "Setpoint": si_sp}

        # Disturbance
        df_d = _load(os.path.join(DATA_DIR, "TIC-100", "HYSYS", "disturbance-tuning", f"TIC_100_{tuning}.CSV"))
        t_d, C_d, R_d = (_arr(df_d, c) for c in ("Time", "TIC-100 - PV", "TIC-100 - SP"))
        si_dist = _si(t_d, C_d, 50.0, y0=50.0, yf=50.0, step_time=TIC_DIST_STEP_TIME, end_time=TIC_DIST_END_TIME, thresh=THRESH_TIC_DIST)
        tic_dist_hy_si[tuning] = si_dist
        tic_dist_hy_sp_si[tuning] = _si(t_d, C_d, R_d, TIC_SP_INITIAL, TIC_SP_FINAL, TIC_SP_STEP_TIME, thresh=THRESH_TIC)
        print(f"\n[HYSYS  {tuning}  disturbance]")
        print(si_dist)
        registry[("TIC-100", "HYSYS", f"Disturbance-{tuning}")] = si_dist

    return {
        "registry": registry,
        "si_fic100_py": si_fic100_py,
        "si_fic100_hy": si_fic100_hy,
        "si_fic101_py": si_fic101_py,
        "si_fic101_hy": si_fic101_hy,
        "si_fic102_py": si_fic102_py,
        "si_fic102_hy": si_fic102_hy,
        "lic_sp_si": lic_sp_si,
        "lic_sp_hy_si": lic_sp_hy_si,
        "lic_dist_si": lic_dist_si,
        "lic_dist_hy_si": lic_dist_hy_si,
        "tic_sp_py_si": tic_sp_py_si,
        "tic_dist_py_si": tic_dist_py_si,
        "tic_dist_py_sp_si": tic_dist_py_sp_si,
        "si_nested": si_nested,
        "tic_sp_hy_si": tic_sp_hy_si,
        "tic_dist_hy_si": tic_dist_hy_si,
        "tic_dist_hy_sp_si": tic_dist_hy_sp_si,
        "si_nested_hy": si_nested_hy,
    }


def main() -> None:
    """Orchestrate performance calculations, error metrics evaluations, validation logging, and plot generations."""
    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "reports", "validation.log")
    
    # Open validation log file
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("Validation Log - Biodiesel Reactor Control Simulation\n")
        lf.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        lf.write("=" * 80 + "\n\n")

    def _log_warn(msg: str):
        print(f"WARNING: {msg}")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"WARNING: {msg}\n")
            
    def _log_info(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{msg}\n")

    # 1. Execute original StepInfo workflow
    _log_info("Executing StepInfo dynamic response metrics calculations (silently)...")
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        results = calculate_all_performance()
    
    _write_summary(
        registry=results["registry"],
        txt_path=os.path.join(OUTPUT_DIR, "reports", "summary.txt"),
        csv_path=os.path.join(OUTPUT_DIR, "data", "summary.csv"),
    )
    _log_info("Original summary report files written successfully.\n")

    # 2. Run new validation metrics and error analysis workflow
    _log_info("Executing mathematical validation metrics comparisons...")
    
    validation_results = []
    residual_rows = []
    
    # Iterate through all validation cases
    for case in VALIDATION_CASES:
        loop = case["loop"]
        scenario = case["scenario"]
        py_path = case["py_path"]
        hy_path = case["hy_path"]
        
        _log_info(f"Processing Case: {loop} - {scenario}...")
        
        # Load CSVs
        df_py = load_csv(py_path)
        df_hy = load_csv(hy_path)
        
        # Align time-series
        t_ref, df_py_aligned, df_hy_aligned = align_time_series(df_py, df_hy, "Time")
        
        # Extract variables dynamically according to COLUMN_MAPPINGS
        mappings = COLUMN_MAPPINGS.get(loop, {})
        for var in VALIDATION_VARIABLES:
            if var not in mappings:
                continue
                
            py_col = mappings[var]["py"]
            hy_col = mappings[var]["hy"]
            
            # Verify columns
            if py_col not in df_py_aligned.columns:
                raise KeyError(
                    f"Required column '{py_col}' is missing in Python CSV file '{py_path}'.\n"
                    f"Available columns: {list(df_py.columns)}"
                )
            if hy_col not in df_hy_aligned.columns:
                raise KeyError(
                    f"Required column '{hy_col}' is missing in HYSYS CSV file '{hy_path}'.\n"
                    f"Available columns: {list(df_hy.columns)}"
                )
                
            y_py = np.asarray(df_py_aligned[py_col].values, dtype=float)
            y_hy = np.asarray(df_hy_aligned[hy_col].values, dtype=float)
            
            # Calculate validation metrics
            metrics = calculate_validation_metrics(y_hy, y_py)
            metrics["loop"] = loop
            metrics["scenario"] = scenario
            metrics["variable"] = var
            
            # Print and log warnings if any
            if metrics["Warnings"]:
                for warn in metrics["Warnings"]:
                    _log_warn(f"[{loop} - {scenario} - {var}] {warn}")
                    
            validation_results.append(metrics)
            
            # Calculate residual time-series data
            errors = y_py - y_hy
            abs_errors = np.abs(errors)
            for i in range(len(t_ref)):
                residual_rows.append({
                    "Time": t_ref[i],
                    "Loop": loop,
                    "Scenario": scenario,
                    "Variable": var,
                    "HYSYS": y_hy[i],
                    "Python": y_py[i],
                    "Error": errors[i],
                    "AbsoluteError": abs_errors[i]
                })
                

            
    # Save residual data CSV
    residual_path = os.path.join(OUTPUT_DIR, "data", "residual_data.csv")
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)
    _log_info(f"Residual time-series dataset saved -> {residual_path}")
    
    # Save validation metrics JSON
    json_path = os.path.join(OUTPUT_DIR, "reports", "validation_metrics.json")
    json_data = {}
    for r in validation_results:
        key = f"{r['loop']}_{r['scenario']}_{r['variable']}"
        json_data[key] = {k: v for k, v in r.items() if k not in ("loop", "scenario", "variable", "Warnings")}
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, indent=2)
    _log_info(f"Structured metrics JSON dataset saved -> {json_path}")
    
    # Save validation metrics CSV
    csv_rows = []
    for r in validation_results:
        csv_rows.append({
            "Loop": r["loop"],
            "Scenario": r["scenario"],
            "Variable": r["variable"],
            "N": r["N"],
            "MeanHYSYS": r["MeanHYSYS"],
            "MeanPython": r["MeanPython"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "MaxAbsErr": r["MaxAbsErr"],
            "MAPE": r["MAPE"],
            "ExcludedMAPE": r["ExcludedMAPE"],
            "SMAPE": r["SMAPE"],
            "ExcludedSMAPE": r["ExcludedSMAPE"],
            "MBE": r["MBE"],
            "PBIAS": r["PBIAS"],
            "R2": r["R2"]
        })
    csv_path = os.path.join(OUTPUT_DIR, "data", "validation_summary.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    _log_info(f"Validation summary table saved -> {csv_path}")
    
    # Save human-readable validation summary txt
    txt_path = os.path.join(OUTPUT_DIR, "reports", "validation_summary.txt")
    write_summary(validation_results, txt_path)
    
    # 3. Add Control Loop Performance Integral Errors log
    _log_info("\nCalculating Control Loop Performance Integral Errors (SP vs PV)...")
    integral_logs = ["", "=" * 80, "  CONTROL LOOP INTEGRAL PERFORMANCE ERRORS (SP vs PV)", "=" * 80]
    
    for case in VALIDATION_CASES:
        loop = case["loop"]
        scenario = case["scenario"]
        
        # Load and align
        df_py = load_csv(case["py_path"])
        df_hy = load_csv(case["hy_path"])
        t_ref, df_py_aligned, df_hy_aligned = align_time_series(df_py, df_hy, "Time")
        
        mappings = COLUMN_MAPPINGS.get(loop, {})
        py_pv_col = mappings["PV"]["py"]
        py_sp_col = mappings["SP"]["py"]
        hy_pv_col = mappings["PV"]["hy"]
        hy_sp_col = mappings["SP"]["hy"]
        
        # Python SP vs PV
        err_py = np.asarray(df_py_aligned[py_sp_col].values, dtype=float) - np.asarray(df_py_aligned[py_pv_col].values, dtype=float)
        int_py = calculate_integral_error(t_ref, err_py)
        
        # HYSYS SP vs PV
        err_hy = np.asarray(df_hy_aligned[hy_sp_col].values, dtype=float) - np.asarray(df_hy_aligned[hy_pv_col].values, dtype=float)
        int_hy = calculate_integral_error(t_ref, err_hy)
        
        integral_logs.append(f"\n  [ {loop} - {scenario} ]")
        integral_logs.append(f"    Python: IAE = {int_py['IAE']:12.2f} | ISE = {int_py['ISE']:12.2f} | ITAE = {int_py['ITAE']:12.2f}")
        integral_logs.append(f"    HYSYS : IAE = {int_hy['IAE']:12.2f} | ISE = {int_hy['ISE']:12.2f} | ITAE = {int_hy['ITAE']:12.2f}")
        
    integral_logs.append("=" * 80 + "\n")
    integral_txt = "\n".join(integral_logs)
    print(integral_txt)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(integral_txt)
        
    _log_info("Validation task completed successfully.\n")
    
    # Display the final validation summary report directly to the terminal
    with open(txt_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
