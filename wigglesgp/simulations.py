import csv
from pathlib import Path

import numpy as np
import yaml

from .fitting import fit_damping_scale, sigma_errors_delta_chi2
from .io_utils import read_power_ratio_snapshot
from .variance import corrected_per_k_variance, smooth_log_variance

def load_simulation_config(path):
    """
    Load a YAML config for fitting Sigma from paired simulations.
    """
    with open(path, "r") as handle:
        return yaml.safe_load(handle)
    
def _read_ratio_pairs(pairs, box_size, k_min, k_max, numerator_key="wiggle", denominator_key="vanilla"):
    """
    Read paired power-spectrum files and return a common k-grid plus ratios.

    Returns
    -------
    k_ref: ndarray
        Common k-grid.
    ratios : ndarray, shape (n_pairs, n_k)
        Power-spectrum ratios for each paired simulation
    """
    k_ref = None
    ratios = []

    for pair in pairs:
        k, ratio = read_power_ratio_snapshot(
            pair[numerator_key],
            pair[denominator_key],
            box_size=box_size,
            k_min=k_min,
            k_max=k_max,
        )

        if k_ref is None:
            k_ref = k
        elif len(k) != len(k_ref):
            raise ValueError(
                "All files in a ratio set must have the same number of k-bins"
            )
        
        ratios.append(ratio)

    if len(ratios) == 0:
        raise ValueError("At least one ratio pair is required")

    return k_ref, np.asarray(ratios, dtype=float)

def fit_sigma_table_from_config(config_path, write=True):
    """
    Fit Sigma(z, omega) values from paired wiggle/vanilla simulations.

    Each dataset in the YAML config should correspond to one (z, omega)
    point and contain:

    signal_pairs:
        Wiggle/Vanilla ratios used in the damping-envelope fit.

    variance_pairs:
        Independent wiggle/vanilla ratios witht eh same feature parameters,
        used to esitmate the per-k simulation varaince

    Returns
    -------
    rows: list of dict
        One row per fitted (z, omega) point.
    """

    config = load_simulation_config(config_path)

    feature_type = config.get("feature_type", None)
    if feature_type not in ["log", "linear"]:
        raise ValueError(f"{feature_type!r} is not a valid feature type. "
                         "Config file must specify feature_type of either 'log' or 'linear'."
                         )
    sim_cfg = config.get("simulation", {})
    box_size = sim_cfg.get("box_size", 1024.0)
    k_min = sim_cfg.get("k_min", 0.05)
    k_max = sim_cfg.get("k_max", 0.6)
    h = sim_cfg.get("h", 0.67)

    feature_cfg = config.get("feature_defaults", {})
    amplitude = feature_cfg.get("amplitude", 0.03)
    phase = feature_cfg.get("phase", 0.0)
    k_pivot = feature_cfg.get("k_pivot", 0.05)
    phase_free = feature_cfg.get("phase_free", False)

    rows = []

    for dataset in config["datasets"]:
        z = float(dataset["z"])
        omega_label = float(dataset["omega"])
        omega_model = 10.0 ** omega_label

        k_signal, signal_ratios = _read_ratio_pairs(
            dataset["signal_pairs"],
            box_size=box_size,
            k_min=k_min,
            k_max=k_max,
        )

        k_var, variance_ratios = _read_ratio_pairs(
            dataset["variance_pairs"],
            box_size=box_size,
            k_min=k_min,
            k_max=k_max,
        )

        
        if len(k_signal) != len(k_var):
            raise ValueError(
                "Signal ratios and variance ratios must have the same number of k-bins"
            )

        variance_raw = corrected_per_k_variance(variance_ratios)
        variance = smooth_log_variance(k_var, variance_raw, degree=3)

        # print("\nDEBUG")
        # print(f"z={z}, omega={omega_model}, omega_label={omega_label}")
        # print("signal_ratios shape:", signal_ratios.shape)
        # print("variance_ratios shape:", variance_ratios.shape)
        # print("signal min/max:", np.nanmin(signal_ratios), np.nanmax(signal_ratios))
        # print("variance ratio min/max:", np.nanmin(variance_ratios), np.nanmax(variance_ratios))
        # print("variance min/median/max:", np.nanmin(variance), np.nanmedian(variance), np.nanmax(variance))
        # print("std min/median/max:", np.nanmin(np.sqrt(variance)), np.nanmedian(np.sqrt(variance)), np.nanmax(np.sqrt(variance)))

        fit = fit_damping_scale(
            k=k_signal,
            y=signal_ratios,
            omega=omega_model,
            variance=variance,
            phase_seed=phase,
            phase_free=phase_free,
            feature_type=feature_type,
            amplitude=amplitude,
            k_pivot=k_pivot,
            h=h,
        )

        dpos, dneg, dsym = sigma_errors_delta_chi2(
            omega=omega_model,
            k=k_signal,
            y=signal_ratios,
            std=fit["std"],
            sigma_hat=fit["sigma"],
            phase_hat=fit["phase"],
            phase_free=phase_free,
            feature_type=feature_type,
            amplitude=amplitude,
            k_pivot=k_pivot,
            h=h,
        )

        rows.append(
            {
                "feature_type": feature_type,
                "z": z,
                "omega": omega_label,
                "omega_model": omega_model,
                "sigma": fit["sigma"],
                "variance_model": "log_poly3",
                "sigma_err": dsym,
                "sigma_err_plus": dpos,
                "sigma_err_minus": dneg,
                "phase": fit["phase"],
                "chi2": fit["chi2"],
                "chi2_red": fit["chi2_red"],
                "n_signal_pairs": int(signal_ratios.shape[0]),
                "n_variance_pairs": int(variance_ratios.shape[0]),
                "k_min": float(np.min(k_signal)),
                "k_max": float(np.max(k_signal)),
            }
        )

    output_cfg = config.get("output", {})
    output_path = output_cfg.get("sigma_table")

    if write and output_path is not None:
        write_sigma_table(rows, output_path)
        
    return rows

def write_sigma_table(rows, path):
    """
    Write a fitted Sigma table to CSV.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        raise ValueError("Cannot write an empty Sigma table.")
    
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)