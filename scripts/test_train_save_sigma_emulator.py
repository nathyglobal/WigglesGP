from pathlib import Path

import numpy as np
import yaml

from wigglesgp.simulations import fit_sigma_table_from_config
from wigglesgp.emulator import SigmaEmulator


def write_log_config(path):
    data_root = Path("/Volumes/NAS/Research Backup/Data/DataV2")
    variance_root = Path("/Volumes/NAS/Research Backup/Data/Variance Data")

    omegas = ["0.8", "0.87", "1.26", "1.5", "2.0"]
    snapshots = ["000", "001", "002", "003", "004", "005"]

    config = {
        "feature_type": "log",
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
            "sigma_table": "training_data/log_sigma_fits.csv",
        },
    }

    for omega in omegas:
        omega_dir = data_root / f"Log{omega}"

        for snap in snapshots:
            z = 5.0 - float(int(snap))

            signal_pairs = [
                {
                    "wiggle": str(
                        omega_dir / f"PK-DM-A_003_logF_{omega}snapshotP_{snap}"
                    ),
                    "vanilla": str(
                        omega_dir / f"PK-DM-A_003_logF_{omega}snapshotNP_{snap}"
                    ),
                }
            ]

            variance_pairs = []
            for seed in [1, 2, 3, 4]:
                variance_pairs.append(
                    {
                        "wiggle": str(
                            variance_root
                            / f"PK-DM-Set1_1-{seed}_N1024_L1024_P_{snap}"
                        ),
                        "vanilla": str(
                            variance_root
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return path


def check_paths_exist(config_path):
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)

    missing = []

    for dataset in config["datasets"]:
        for group in ["signal_pairs", "variance_pairs"]:
            for pair in dataset[group]:
                for key in ["wiggle", "vanilla"]:
                    path = Path(pair[key])
                    if not path.exists():
                        missing.append(str(path))

    if missing:
        print(f"Missing {len(missing)} files. First few:")
        for path in missing[:10]:
            print(f"  {path}")
        raise FileNotFoundError("Some files in the generated YAML do not exist.")

    print("All YAML file paths exist.")


def print_fit_summary(rows):
    print("\nSigma fit summary")
    print("-----------------")

    for row in rows:
        print(
            f"z={row['z']:.1f}, "
            f"omega={row['omega']:.2f}, "
            f"sigma={row['sigma']:.6g}, "
            f"sigma_err={row['sigma_err']:.6g}, "
            f"chi2_red={row['chi2_red']:.6g}"
        )


def main():
    config_path = write_log_config("configs/log_simulations.yaml")
    print(f"Wrote config: {config_path}")

    check_paths_exist(config_path)

    rows = fit_sigma_table_from_config(config_path, write=True)
    print_fit_summary(rows)

    sigma_table = "training_data/log_sigma_fits.csv"

    emulator = SigmaEmulator.from_sigma_table(
        sigma_table,
        feature_type="log",
        normalise_y=True,
        n_restarts_optimizer=10,
        h=0.67,
    )

    print("\nTrained GP metadata")
    print("-------------------")
    for key, value in emulator.metadata.items():
        print(f"{key}: {value}")

    emulator_path = Path("emulators/log_sigma_gp.pkl")
    emulator_path.parent.mkdir(parents=True, exist_ok=True)
    emulator.to_file(emulator_path)
    print(f"\nSaved emulator: {emulator_path}")

    loaded = SigmaEmulator.from_file(emulator_path)

    z_test = np.array([0.0, 1.0, 2.5, 5.0])
    omega_test = np.array([0.8, 1.0, 1.5, 2.0])

    sigma_mean, sigma_std = loaded.predict_sigma(
        z_test,
        omega_test,
        return_std=True,
    )

    print("\nGP Sigma predictions")
    print("--------------------")
    for z, omega, mean, std in zip(z_test, omega_test, sigma_mean, sigma_std):
        print(
            f"z={z:.2f}, omega={omega:.2f}: "
            f"sigma={mean:.6g} ± {std:.6g}"
        )

    k_test = np.linspace(0.05, 0.6, 8)
    z_grid = np.full_like(k_test, 1.0)
    omega_grid = np.full_like(k_test, 1.26)

    damping, sigma = loaded.damping(
        k_test,
        z_grid,
        omega_grid,
        return_sigma=True,
    )

    print("\nDamping prediction at z=1, omega=1.26")
    print("-------------------------------------")
    for k, sig, damp in zip(k_test, sigma, damping):
        print(f"k={k:.4f}, sigma={sig:.6g}, D={damp:.6g}")

    print("\nDone.")


if __name__ == "__main__":
    main()