#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.training import load_sigma_table


DEFAULT_PATHS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "sigma_table": Path("training_data/log_sigma_fits.csv"),
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "sigma_table": Path("training_data/linear_sigma_fits.csv"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a saved Sigma emulator and compare it to its training table."
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature type to test.",
    )

    parser.add_argument(
        "--emulator",
        type=Path,
        default=None,
        help="Path to saved emulator pickle. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--sigma-table",
        type=Path,
        default=None,
        help="Path to Sigma training table. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def resolve_paths(args):
    defaults = DEFAULT_PATHS[args.feature_type]

    emulator_path = args.emulator or defaults["emulator"]
    sigma_table = args.sigma_table or defaults["sigma_table"]

    return emulator_path, sigma_table


def print_metadata(emulator):
    print("\nLoaded emulator")
    print("---------------")
    print(f"h: {emulator.h}")

    print("\nMetadata")
    print("--------")
    for key, value in emulator.metadata.items():
        print(f"{key}: {value}")


def compare_training_table(emulator, sigma_table, feature_type, check_domain=True):
    z, omega, sigma_table_values, sigma_table_err = load_sigma_table(
        sigma_table,
        feature_type=feature_type,
    )

    sigma_gp, sigma_gp_std = emulator.predict_sigma(
        z,
        omega,
        return_std=True,
        check_domain=check_domain,
    )

    diff = sigma_gp - sigma_table_values
    frac_diff = diff / sigma_table_values

    print("\nTraining-table comparison")
    print("-------------------------")
    print(f"table: {sigma_table}")
    print(f"feature_type: {feature_type}")
    print(f"n points: {len(z)}")

    print("\nAbsolute Sigma difference")
    print("-------------------------")
    print(f"min:    {np.nanmin(np.abs(diff)):.6g}")
    print(f"median: {np.nanmedian(np.abs(diff)):.6g}")
    print(f"max:    {np.nanmax(np.abs(diff)):.6g}")

    print("\nFractional Sigma difference")
    print("---------------------------")
    print(f"min:    {np.nanmin(np.abs(frac_diff)):.6g}")
    print(f"median: {np.nanmedian(np.abs(frac_diff)):.6g}")
    print(f"max:    {np.nanmax(np.abs(frac_diff)):.6g}")

    print("\nGP predictive standard deviation")
    print("-------------------------------")
    print(f"min:    {np.nanmin(sigma_gp_std):.6g}")
    print(f"median: {np.nanmedian(sigma_gp_std):.6g}")
    print(f"max:    {np.nanmax(sigma_gp_std):.6g}")

    print("\nWorst 10 fractional differences")
    print("-------------------------------")

    order = np.argsort(np.abs(frac_diff))[::-1]

    for idx in order[:10]:
        print(
            f"z={z[idx]:4.1f}, "
            f"omega_label={omega[idx]:6.3g}, "
            f"sigma_table={sigma_table_values[idx]:10.6g}, "
            f"sigma_gp={sigma_gp[idx]:10.6g}, "
            f"diff={diff[idx]:+10.4e}, "
            f"frac={frac_diff[idx]:+10.4e}, "
            f"gp_std={sigma_gp_std[idx]:10.6g}, "
            f"table_err={sigma_table_err[idx]:10.6g}"
        )

    return z, omega, sigma_table_values, sigma_gp, sigma_gp_std


def damping_sanity_check(emulator, z, omega, check_domain=True):
    """
    Simple vectorised damping-envelope check using one representative
    table point.
    """
    k = np.linspace(0.05, 0.6, 8)

    z_grid = np.full_like(k, float(z))
    omega_grid = np.full_like(k, float(omega))

    damping, sigma = emulator.damping(
        k,
        z_grid,
        omega_grid,
        return_sigma=True,
        check_domain=check_domain,
    )

    print("\nDamping-envelope sanity check")
    print("-----------------------------")
    print(f"using z={float(z):.1f}, omega_label={float(omega):.3g}")
    print(f"k min/max:       {np.min(k):.6g} / {np.max(k):.6g}")
    print(f"sigma min/max:   {np.min(sigma):.6g} / {np.max(sigma):.6g}")
    print(f"D min/max:       {np.min(damping):.6g} / {np.max(damping):.6g}")

    print("\nDamping samples")
    print("---------------")
    for kk, ss, dd in zip(k, sigma, damping):
        print(f"k={kk:.4f}, sigma={ss:.6g}, D={dd:.6g}")


def main():
    args = parse_args()

    emulator_path, sigma_table = resolve_paths(args)
    check_domain = not args.no_domain_check

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    if not sigma_table.exists():
        raise FileNotFoundError(f"Sigma table does not exist: {sigma_table}")

    emulator = SigmaEmulator.from_file(emulator_path)

    print_metadata(emulator)

    z, omega, sigma_table_values, sigma_gp, sigma_gp_std = compare_training_table(
        emulator,
        sigma_table,
        args.feature_type,
        check_domain=check_domain,
    )

    idx = len(z) // 2
    damping_sanity_check(
        emulator,
        z=z[idx],
        omega=omega[idx],
        check_domain=check_domain,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()