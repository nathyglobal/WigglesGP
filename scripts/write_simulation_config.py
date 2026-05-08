#!/usr/bin/env python3

from pathlib import Path
import argparse

import yaml


FEATURE_SPECS = {
    "log": {
        "directory_prefix": "Log",
        "filename_tag": "logF",
        "omegas": ["0.8", "0.87", "1.26", "1.5", "2.0"],
        "sigma_table": "training_data/log_sigma_fits.csv",
        "config_path": "configs/log_simulations.yaml",
    },
    "linear": {
        "directory_prefix": "Lin",
        "filename_tag": "linF",
        "omegas": ["0.4", "0.8", "0.87", "1.0", "1.2"],
        "sigma_table": "training_data/linear_sigma_fits.csv",
        "config_path": "configs/linear_simulations.yaml",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write simulation YAML config for log or linear feature fits."
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature template to generate.",
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
        "--output",
        type=Path,
        default=None,
        help="Output YAML path. Defaults to configs/<feature>_simulations.yaml.",
    )

    parser.add_argument(
        "--sigma-table",
        type=Path,
        default=None,
        help="Output sigma table path written into the YAML.",
    )

    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Check that all referenced files exist after writing the YAML.",
    )

    return parser.parse_args()


def build_simulation_config(
    *,
    feature_type,
    data_root,
    variance_root,
    sigma_table=None,
):
    spec = FEATURE_SPECS[feature_type]

    snapshots = ["000", "001", "002", "003", "004", "005"]

    if sigma_table is None:
        sigma_table = spec["sigma_table"]

    config = {
        "feature_type": feature_type,
        "simulation": {
            "box_size": 1024.0,
            "k_min": 0.05,
            "k_max": 0.6,
            "h": 0.67,
        },
        "feature_defaults": {
            "amplitude": 0.03,
            "phase": 0.0,
            "k_pivot": 0.05,
            "phase_free": False,
        },
        "datasets": [],
        "output": {
            "sigma_table": str(sigma_table),
        },
    }

    directory_prefix = spec["directory_prefix"]
    filename_tag = spec["filename_tag"]

    for omega in spec["omegas"]:
        omega_dir = Path(data_root) / f"{directory_prefix}{omega}"

        for snap in snapshots:
            z = 5.0 - float(int(snap))

            signal_pairs = [
                {
                    "wiggle": str(
                        omega_dir / f"PK-DM-A_003_{filename_tag}_{omega}snapshotP_{snap}"
                    ),
                    "vanilla": str(
                        omega_dir / f"PK-DM-A_003_{filename_tag}_{omega}snapshotNP_{snap}"
                    ),
                }
            ]

            variance_pairs = []
            for seed in [1, 2, 3, 4]:
                variance_pairs.append(
                    {
                        "wiggle": str(
                            Path(variance_root)
                            / f"PK-DM-Set1_1-{seed}_N1024_L1024_P_{snap}"
                        ),
                        "vanilla": str(
                            Path(variance_root)
                            / f"PK-DM-Set1_2-{seed}_N1024_L1024_NP_{snap}"
                        ),
                    }
                )

            config["datasets"].append(
                {
                    "z": z,
                    "omega": float(omega),
                    "signal_pairs": signal_pairs,
                    "variance_pairs": variance_pairs,
                }
            )

    return config


def iter_config_paths(config):
    for dataset in config["datasets"]:
        for group in ["signal_pairs", "variance_pairs"]:
            for pair in dataset[group]:
                for key in ["wiggle", "vanilla"]:
                    yield Path(pair[key])


def check_paths_exist(config):
    missing = []

    for path in iter_config_paths(config):
        if not path.exists():
            missing.append(path)

    if not missing:
        print("All YAML file paths exist.")
        return

    print(f"Missing {len(missing)} files. First few:")
    for path in missing[:20]:
        print(f"  {path}")

    raise FileNotFoundError("Some files referenced by the generated YAML do not exist.")


def main():
    args = parse_args()

    spec = FEATURE_SPECS[args.feature_type]

    output = args.output
    if output is None:
        output = Path(spec["config_path"])

    config = build_simulation_config(
        feature_type=args.feature_type,
        data_root=args.data_root,
        variance_root=args.variance_root,
        sigma_table=args.sigma_table,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"Wrote config: {output}")
    print(f"feature_type: {args.feature_type}")
    print(f"n datasets:   {len(config['datasets'])}")
    print(f"sigma_table:  {config['output']['sigma_table']}")

    if args.check_paths:
        check_paths_exist(config)


if __name__ == "__main__":
    main()