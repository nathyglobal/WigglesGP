#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import yaml

from wigglesgp.simulations import fit_sigma_table_from_config
from wigglesgp.emulator import SigmaEmulator

# Import shared config-generation helper from scripts directory.
try:
    from write_simulation_config import (
        FEATURE_SPECS,
        build_simulation_config,
        check_paths_exist as check_config_paths_exist,
    )
except ImportError:
    # Allows running from repo root with PYTHONPATH=.
    import sys

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))

    from write_simulation_config import (
        FEATURE_SPECS,
        build_simulation_config,
        check_paths_exist as check_config_paths_exist,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit Sigma values from simulation config, train a Sigma emulator, "
            "save it, reload it, and run simple sanity checks."
        )
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature template to fit and train.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Existing simulation YAML config. If omitted, a config is generated "
            "using scripts/write_simulation_config.py defaults."
        ),
    )

    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write/update the generated config before fitting.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Volumes/NAS/Research Backup/Data/DataV2"),
        help="Root directory containing Log*/Lin* simulation folders.",
    )

    parser.add_argument(
        "--variance-root",
        type=Path,
        default=Path("/Volumes/NAS/Research Backup/Data/Variance Data"),
        help="Root directory containing variance simulation files.",
    )

    parser.add_argument(
        "--sigma-table",
        type=Path,
        default=None,
        help="Output sigma-fit CSV table. Defaults by feature type.",
    )

    parser.add_argument(
        "--emulator-output",
        type=Path,
        default=None,
        help="Output emulator pickle. Defaults to emulators/<feature>_sigma_gp.pkl.",
    )

    parser.add_argument(
        "--normalise-y",
        action="store_true",
        default=True,
        help="Normalise GP target values before training.",
    )

    parser.add_argument(
        "--no-normalise-y",
        action="store_false",
        dest="normalise_y",
        help="Disable GP target normalisation.",
    )

    parser.add_argument(
        "--n-restarts-optimizer",
        type=int,
        default=10,
        help="Number of GP hyperparameter optimiser restarts.",
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0.67,
        help="Dimensionless Hubble parameter used by the damping emulator.",
    )

    parser.add_argument(
        "--check-paths",
        action="store_true",
        default=True,
        help="Check all YAML file paths exist before fitting.",
    )

    parser.add_argument(
        "--no-check-paths",
        action="store_false",
        dest="check_paths",
        help="Skip path existence checks.",
    )

    return parser.parse_args()


def default_config_path(feature_type):
    return Path(FEATURE_SPECS[feature_type]["config_path"])


def default_sigma_table(feature_type):
    return Path(FEATURE_SPECS[feature_type]["sigma_table"])


def default_emulator_path(feature_type):
    return Path("emulators") / f"{feature_type}_sigma_gp.pkl"


def write_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return path


def load_config(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def get_or_write_config(args):
    config_path = args.config or default_config_path(args.feature_type)
    sigma_table = args.sigma_table or default_sigma_table(args.feature_type)

    if args.write_config or not config_path.exists():
        config = build_simulation_config(
            feature_type=args.feature_type,
            data_root=args.data_root,
            variance_root=args.variance_root,
            sigma_table=sigma_table,
        )

        write_config(config_path, config)
        print(f"Wrote config: {config_path}")
    else:
        config = load_config(config_path)
        print(f"Using existing config: {config_path}")

    return config_path, config


def print_fit_summary(rows):
    print("\nSigma fit summary")
    print("-----------------")

    for row in rows:
        print(
            f"z={row['z']:.1f}, "
            f"omega={row['omega']:.3g}, "
            f"sigma={row['sigma']:.6g}, "
            f"sigma_err={row['sigma_err']:.6g}, "
            f"chi2_red={row['chi2_red']:.6g}"
        )


def print_metadata(emulator):
    print("\nTrained GP metadata")
    print("-------------------")

    for key, value in emulator.metadata.items():
        print(f"{key}: {value}")


def prediction_test_points(emulator):
    metadata = emulator.metadata

    z_min = float(metadata.get("z_min", 0.0))
    z_max = float(metadata.get("z_max", 5.0))
    omega_min = float(metadata.get("omega_min", 0.0))
    omega_max = float(metadata.get("omega_max", 1.0))

    z_test = np.array(
        [
            z_min,
            0.5 * (z_min + z_max),
            z_max,
        ],
        dtype=float,
    )

    omega_test = np.array(
        [
            omega_min,
            0.5 * (omega_min + omega_max),
            omega_max,
        ],
        dtype=float,
    )

    return z_test, omega_test


def run_loaded_emulator_checks(emulator):
    z_test, omega_test = prediction_test_points(emulator)

    sigma_mean, sigma_std = emulator.predict_sigma(
        z_test,
        omega_test,
        return_std=True,
    )

    print("\nGP Sigma predictions")
    print("--------------------")
    for z, omega, mean, std in zip(z_test, omega_test, sigma_mean, sigma_std):
        print(
            f"z={z:.2f}, omega={omega:.3g}: "
            f"sigma={mean:.6g} ± {std:.6g}"
        )

    k_test = np.linspace(0.05, 0.6, 8)

    z_mid = np.full_like(k_test, z_test[len(z_test) // 2])
    omega_mid = np.full_like(k_test, omega_test[len(omega_test) // 2])

    damping, sigma = emulator.damping(
        k_test,
        z_mid,
        omega_mid,
        return_sigma=True,
    )

    print(
        f"\nDamping prediction at z={z_mid[0]:.3g}, "
        f"omega={omega_mid[0]:.3g}"
    )
    print("-------------------------------------")
    for k, sig, damp in zip(k_test, sigma, damping):
        print(f"k={k:.4f}, sigma={sig:.6g}, D={damp:.6g}")


def main():
    args = parse_args()

    config_path, config = get_or_write_config(args)

    if args.check_paths:
        check_config_paths_exist(config)

    rows = fit_sigma_table_from_config(config_path, write=True)
    print_fit_summary(rows)

    sigma_table = Path(config["output"]["sigma_table"])

    emulator = SigmaEmulator.from_sigma_table(
        sigma_table,
        feature_type=args.feature_type,
        normalise_y=args.normalise_y,
        n_restarts_optimizer=args.n_restarts_optimizer,
        h=args.h,
    )

    print_metadata(emulator)

    emulator_path = args.emulator_output or default_emulator_path(args.feature_type)
    emulator_path.parent.mkdir(parents=True, exist_ok=True)

    emulator.to_file(emulator_path)
    print(f"\nSaved emulator: {emulator_path}")

    loaded = SigmaEmulator.from_file(emulator_path)
    run_loaded_emulator_checks(loaded)

    print("\nDone.")


if __name__ == "__main__":
    main()