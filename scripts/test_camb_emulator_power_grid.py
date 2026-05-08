#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.power import nonlinear_wiggle_power
from wigglesgp.camb_power import get_vanilla_and_wiggle_spectra


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "omega_values": [0.8, 1.26, 2.0],
        "output_dir": Path("forecast_tests/log_camb_emulator_grid"),
        "summary": Path("forecast_tests/log_camb_emulator_grid_summary.csv"),
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "omega_values": [0.4, 0.87, 1.2],
        "output_dir": Path("forecast_tests/linear_camb_emulator_grid"),
        "summary": Path("forecast_tests/linear_camb_emulator_grid_summary.csv"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the CAMB + Sigma-emulator damping test over a small "
            "redshift/frequency grid."
        )
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature template.",
    )

    parser.add_argument(
        "--emulator",
        type=Path,
        default=None,
        help="Path to saved Sigma emulator. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--z-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 5.0],
        help="Redshifts to evaluate.",
    )

    parser.add_argument(
        "--omega-values",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Feature frequency labels in the emulator convention. "
            "Defaults from --feature-type."
        ),
    )

    parser.add_argument(
        "--A-feat",
        type=float,
        default=0.03,
        help="Feature amplitude.",
    )

    parser.add_argument(
        "--phi",
        type=float,
        default=0.0,
        help="Feature phase in radians.",
    )

    parser.add_argument(
        "--kmax",
        type=float,
        default=0.8,
        help="Maximum k/h returned by CAMB.",
    )

    parser.add_argument(
        "--npoints",
        type=int,
        default=700,
        help="Number of CAMB matter-power k points.",
    )

    parser.add_argument(
        "--k-fit-min",
        type=float,
        default=0.05,
        help="Minimum k/h used for emulator-damped diagnostics.",
    )

    parser.add_argument(
        "--k-fit-max",
        type=float,
        default=0.6,
        help="Maximum k/h used for emulator-damped diagnostics.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-grid-point CSV outputs. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to summary CSV. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    defaults = DEFAULTS[args.feature_type]

    emulator_path = args.emulator or defaults["emulator"]
    omega_values = args.omega_values or defaults["omega_values"]
    output_dir = args.output_dir or defaults["output_dir"]
    summary = args.summary or defaults["summary"]

    return emulator_path, omega_values, output_dir, summary


def format_float_for_filename(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def write_output_csv(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "k,"
        "p_van_lin,"
        "p_van_nl,"
        "p_wig_lin,"
        "damping,"
        "sigma,"
        "p_wig_nl_emulator,"
        "ratio_lin,"
        "ratio_nl_emulator"
    )

    array = np.column_stack(
        [
            data["k"],
            data["p_van_lin"],
            data["p_van_nl"],
            data["p_wig_lin"],
            data["damping"],
            data["sigma"],
            data["p_wig_nl"],
            data["ratio_lin"],
            data["ratio_nl"],
        ]
    )

    np.savetxt(path, array, delimiter=",", header=header, comments="")


def run_one_grid_point(
    *,
    emulator,
    feature_type,
    z,
    omega_label,
    A_feat,
    phi,
    kmax,
    npoints,
    k_fit_min,
    k_fit_max,
    check_domain,
    output_path,
):
    spectra = get_vanilla_and_wiggle_spectra(
        redshift=z,
        log10omega_feat=omega_label,
        A_feat=A_feat,
        phi=phi,
        feature_type=feature_type,
        kmax=kmax,
        npoints=npoints,
    )

    k = spectra["k"]
    p_van_lin = spectra["p_van_lin"]
    p_van_nl = spectra["p_van_nl"]
    p_wig_lin = spectra["p_wig_lin"]

    mask = (k >= k_fit_min) & (k <= k_fit_max)

    if not np.any(mask):
        raise ValueError(
            f"No CAMB k values inside requested fit range "
            f"[{k_fit_min}, {k_fit_max}]."
        )

    k_fit = k[mask]
    p_van_lin_fit = p_van_lin[mask]
    p_van_nl_fit = p_van_nl[mask]
    p_wig_lin_fit = p_wig_lin[mask]

    z_grid = np.full_like(k_fit, float(z), dtype=float)
    omega_grid = np.full_like(k_fit, float(omega_label), dtype=float)

    damping, sigma = emulator.damping(
        k_fit,
        z_grid,
        omega_grid,
        return_sigma=True,
        check_domain=check_domain,
    )

    p_wig_nl = nonlinear_wiggle_power(
        p_van_lin=p_van_lin_fit,
        p_van_nl=p_van_nl_fit,
        p_wig_lin=p_wig_lin_fit,
        damping=damping,
    )

    ratio_lin = p_wig_lin_fit / p_van_lin_fit
    ratio_nl = p_wig_nl / p_van_nl_fit
    vanilla_boost = p_van_nl_fit / p_van_lin_fit

    output_data = {
        "k": k_fit,
        "p_van_lin": p_van_lin_fit,
        "p_van_nl": p_van_nl_fit,
        "p_wig_lin": p_wig_lin_fit,
        "damping": damping,
        "sigma": sigma,
        "p_wig_nl": p_wig_nl,
        "ratio_lin": ratio_lin,
        "ratio_nl": ratio_nl,
    }

    write_output_csv(output_path, output_data)

    return {
        "z": float(z),
        "omega_label": float(omega_label),
        "omega_model": 10.0 ** float(omega_label),
        "A_feat": float(A_feat),
        "phi": float(phi),
        "n_k": int(len(k_fit)),
        "k_min": float(np.nanmin(k_fit)),
        "k_max": float(np.nanmax(k_fit)),
        "sigma_min": float(np.nanmin(sigma)),
        "sigma_max": float(np.nanmax(sigma)),
        "damping_min": float(np.nanmin(damping)),
        "damping_max": float(np.nanmax(damping)),
        "ratio_lin_min": float(np.nanmin(ratio_lin)),
        "ratio_lin_max": float(np.nanmax(ratio_lin)),
        "ratio_nl_min": float(np.nanmin(ratio_nl)),
        "ratio_nl_max": float(np.nanmax(ratio_nl)),
        "vanilla_boost_min": float(np.nanmin(vanilla_boost)),
        "vanilla_boost_max": float(np.nanmax(vanilla_boost)),
        "output": str(output_path),
    }


def write_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "z",
        "omega_label",
        "omega_model",
        "A_feat",
        "phi",
        "n_k",
        "k_min",
        "k_max",
        "sigma_min",
        "sigma_max",
        "damping_min",
        "damping_max",
        "ratio_lin_min",
        "ratio_lin_max",
        "ratio_nl_min",
        "ratio_nl_max",
        "vanilla_boost_min",
        "vanilla_boost_max",
        "output",
    ]

    with open(path, "w") as handle:
        handle.write(",".join(columns) + "\n")

        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, str):
                    values.append(value)
                else:
                    values.append(f"{value:.12g}")
            handle.write(",".join(values) + "\n")


def main():
    args = parse_args()
    check_domain = not args.no_domain_check

    emulator_path, omega_values, output_dir, summary_path = resolve_defaults(args)

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    emulator = SigmaEmulator.from_file(emulator_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCAMB + Sigma-emulator grid test")
    print("-------------------------------")
    print(f"emulator:          {emulator_path}")
    print(f"feature_type:      {args.feature_type}")
    print(f"z_values:          {args.z_values}")
    print(f"omega_values:      {omega_values}")
    print(f"A_feat:            {args.A_feat}")
    print(f"phi:               {args.phi}")
    print(f"output_dir:        {output_dir}")
    print(f"summary:           {summary_path}")

    summary_rows = []

    total = len(args.z_values) * len(omega_values)
    counter = 0

    for z in args.z_values:
        for omega_label in omega_values:
            counter += 1

            z_label = format_float_for_filename(z)
            omega_file_label = format_float_for_filename(omega_label)

            output_path = (
                output_dir
                / f"{args.feature_type}_z{z_label}_w{omega_file_label}.csv"
            )

            print(
                f"\n[{counter}/{total}] "
                f"z={z:g}, omega_label={omega_label:g} "
                f"-> {output_path}"
            )

            row = run_one_grid_point(
                emulator=emulator,
                feature_type=args.feature_type,
                z=z,
                omega_label=omega_label,
                A_feat=args.A_feat,
                phi=args.phi,
                kmax=args.kmax,
                npoints=args.npoints,
                k_fit_min=args.k_fit_min,
                k_fit_max=args.k_fit_max,
                check_domain=check_domain,
                output_path=output_path,
            )

            summary_rows.append(row)

            print(
                "  "
                f"sigma={row['sigma_min']:.6g}/{row['sigma_max']:.6g}, "
                f"D={row['damping_min']:.6g}/{row['damping_max']:.6g}, "
                f"Rlin={row['ratio_lin_min']:.6g}/{row['ratio_lin_max']:.6g}, "
                f"Rnl={row['ratio_nl_min']:.6g}/{row['ratio_nl_max']:.6g}"
            )

    write_summary_csv(summary_path, summary_rows)

    print("\nDone.")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()