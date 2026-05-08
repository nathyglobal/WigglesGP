#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.power import nonlinear_wiggle_power
from wigglesgp.camb_power import get_vanilla_and_wiggle_spectra


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the CAMB + Sigma-emulator damping test over a small "
            "redshift/frequency grid."
        )
    )

    parser.add_argument(
        "--emulator",
        type=Path,
        default=Path("emulators/log_sigma_gp.pkl"),
        help="Path to saved Sigma emulator.",
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        default="log",
        help="Feature template.",
    )

    parser.add_argument(
        "--z-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 5.0],
        help="Redshifts to evaluate.",
    )

    parser.add_argument(
        "--log10omega-values",
        type=float,
        nargs="+",
        default=[0.8, 1.26, 2.0],
        help="Feature frequency labels in the emulator convention.",
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
        default=Path("forecast_tests/log_camb_emulator_grid"),
        help="Directory for per-grid-point CSV outputs.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("forecast_tests/log_camb_emulator_grid_summary.csv"),
        help="Path to summary CSV.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


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
    log10omega,
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
        log10omega_feat=log10omega,
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
    omega_grid = np.full_like(k_fit, float(log10omega), dtype=float)

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

    summary = {
        "z": float(z),
        "log10omega": float(log10omega),
        "omega": 10.0 ** float(log10omega),
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

    return summary


def write_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "z",
        "log10omega",
        "omega",
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

    emulator = SigmaEmulator.from_file(args.emulator)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCAMB + Sigma-emulator grid test")
    print("-------------------------------")
    print(f"emulator:          {args.emulator}")
    print(f"feature_type:      {args.feature_type}")
    print(f"z_values:          {args.z_values}")
    print(f"log10omega_values: {args.log10omega_values}")
    print(f"A_feat:            {args.A_feat}")
    print(f"phi:               {args.phi}")
    print(f"output_dir:        {args.output_dir}")
    print(f"summary:           {args.summary}")

    summary_rows = []

    total = len(args.z_values) * len(args.log10omega_values)
    counter = 0

    for z in args.z_values:
        for log10omega in args.log10omega_values:
            counter += 1

            z_label = format_float_for_filename(z)
            omega_label = format_float_for_filename(log10omega)

            output_path = (
                args.output_dir
                / f"{args.feature_type}_z{z_label}_w{omega_label}.csv"
            )

            print(
                f"\n[{counter}/{total}] "
                f"z={z:g}, log10omega={log10omega:g} "
                f"-> {output_path}"
            )

            row = run_one_grid_point(
                emulator=emulator,
                feature_type=args.feature_type,
                z=z,
                log10omega=log10omega,
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

    write_summary_csv(args.summary, summary_rows)

    print("\nDone.")
    print(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()