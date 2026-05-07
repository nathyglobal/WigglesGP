from pathlib import Path
import yaml

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
        z = float(int(snap))

        signal_pairs = [
            {
                "wiggle": str(omega_dir / f"PK-DM-A_003_logF_{omega}snapshotP_{snap}"),
                "vanilla": str(omega_dir / f"PK-DM-A_003_logF_{omega}snapshotNP_{snap}"),
            }
        ]

        variance_pairs = []
        for seed in [1, 2, 3, 4]:
            variance_pairs.append(
                {
                    "wiggle": str(
                        variance_root / f"PK-DM-Set1_1_{seed}_N1024_L1024_P_{snap}"
                    ),
                    "vanilla": str(
                        variance_root / f"PK-DM-Set1_2_{seed}_N1024_L1024_NP_{snap}"
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

Path("configs").mkdir(exist_ok=True)

with open("configs/log_simulations.yaml", "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)