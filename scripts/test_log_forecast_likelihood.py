#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.forecasting import (
    emulator_damped_camb_data_vector,
    diagonal_fractional_sigma,
    gaussian_loglike_diagonal,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple CAMB + Sigma-emulator forecast likelihood "
            "for primordial-feature parameters."
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
        "--redshifts",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 5.0],
        help="Redshift bins in the forecast data vector.",
    )

    parser.add_argument(
        "--fid-log10omega",
        type=float,
        default=1.26,
        help="Fiducial log10 feature frequency.",
    )

    parser.add_argument(
        "--fid-A-feat",
        type=float,
        default=0.03,
        help="Fiducial feature amplitude.",
    )

    parser.add_argument(
        "--fid-phi",
        type=float,
        default=0.0,
        help="Fiducial feature phase in radians.",
    )

    parser.add_argument(
        "--frac-error",
        type=float,
        default=0.01,
        help="Simple fractional error assigned to each data-vector element.",
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
        help="Minimum k/h used in the forecast data vector.",
    )

    parser.add_argument(
        "--k-fit-max",
        type=float,
        default=0.6,
        help="Maximum k/h used in the forecast data vector.",
    )

    parser.add_argument(
        "--observable",
        choices=["ratio_nl", "p_wig_nl"],
        default="ratio_nl",
        help="Forecast observable.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forecast_tests/log_forecast_likelihood_scan.csv"),
        help="Output CSV containing test likelihood evaluations.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def build_test_points(fid_A, fid_log10omega, fid_phi):
    """
    Small deterministic set of likelihood test points.

    This is not a sampler. It is just a redline test that the likelihood
    responds sensibly to A, frequency, and phase.
    """
    return [
        {
            "label": "fiducial",
            "A_feat": fid_A,
            "log10omega": fid_log10omega,
            "phi": fid_phi,
        },
        {
            "label": "lower_A",
            "A_feat": 0.8 * fid_A,
            "log10omega": fid_log10omega,
            "phi": fid_phi,
        },
        {
            "label": "higher_A",
            "A_feat": 1.2 * fid_A,
            "log10omega": fid_log10omega,
            "phi": fid_phi,
        },
        {
            "label": "lower_omega",
            "A_feat": fid_A,
            "log10omega": fid_log10omega - 0.05,
            "phi": fid_phi,
        },
        {
            "label": "higher_omega",
            "A_feat": fid_A,
            "log10omega": fid_log10omega + 0.05,
            "phi": fid_phi,
        },
        {
            "label": "phase_plus_0p25",
            "A_feat": fid_A,
            "log10omega": fid_log10omega,
            "phi": fid_phi + 0.25,
        },
        {
            "label": "phase_plus_0p5",
            "A_feat": fid_A,
            "log10omega": fid_log10omega,
            "phi": fid_phi + 0.5,
        },
    ]


def write_results(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "label",
        "A_feat",
        "log10omega",
        "phi",
        "loglike",
        "chi2",
        "delta_chi2",
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

    print("\nBuilding fiducial forecast data vector")
    print("--------------------------------------")
    print(f"emulator:       {args.emulator}")
    print(f"feature_type:   {args.feature_type}")
    print(f"redshifts:      {args.redshifts}")
    print(f"observable:     {args.observable}")
    print(f"fid A_feat:     {args.fid_A_feat}")
    print(f"fid log10omega: {args.fid_log10omega}")
    print(f"fid phi:        {args.fid_phi}")
    print(f"frac_error:     {args.frac_error}")

    fid = emulator_damped_camb_data_vector(
        emulator=emulator,
        redshifts=args.redshifts,
        log10omega=args.fid_log10omega,
        A_feat=args.fid_A_feat,
        phi=args.fid_phi,
        feature_type=args.feature_type,
        kmax=args.kmax,
        npoints=args.npoints,
        k_fit_min=args.k_fit_min,
        k_fit_max=args.k_fit_max,
        check_domain=check_domain,
        observable=args.observable,
    )

    data = fid["vector"]
    sigma = diagonal_fractional_sigma(data, args.frac_error)

    print(f"data-vector length: {len(data)}")
    print(f"k range:            {np.nanmin(fid['k']):.6g} -> {np.nanmax(fid['k']):.6g}")
    print(f"z range:            {np.nanmin(fid['z']):.6g} -> {np.nanmax(fid['z']):.6g}")

    test_points = build_test_points(
        args.fid_A_feat,
        args.fid_log10omega,
        args.fid_phi,
    )

    rows = []

    print("\nLikelihood test points")
    print("----------------------")

    fid_chi2 = None

    for point in test_points:
        model = emulator_damped_camb_data_vector(
            emulator=emulator,
            redshifts=args.redshifts,
            log10omega=point["log10omega"],
            A_feat=point["A_feat"],
            phi=point["phi"],
            feature_type=args.feature_type,
            kmax=args.kmax,
            npoints=args.npoints,
            k_fit_min=args.k_fit_min,
            k_fit_max=args.k_fit_max,
            check_domain=check_domain,
            observable=args.observable,
        )

        loglike, chi2 = gaussian_loglike_diagonal(
            model["vector"],
            data,
            sigma,
        )

        if point["label"] == "fiducial":
            fid_chi2 = chi2

        delta_chi2 = chi2 - fid_chi2 if fid_chi2 is not None else np.nan

        row = {
            "label": point["label"],
            "A_feat": point["A_feat"],
            "log10omega": point["log10omega"],
            "phi": point["phi"],
            "loglike": loglike,
            "chi2": chi2,
            "delta_chi2": delta_chi2,
        }

        rows.append(row)

        print(
            f"{point['label']:>16s}: "
            f"A={point['A_feat']:.5g}, "
            f"log10omega={point['log10omega']:.5g}, "
            f"phi={point['phi']:.5g}, "
            f"chi2={chi2:.6g}, "
            f"Delta chi2={delta_chi2:.6g}"
        )

    write_results(args.output, rows)

    print("\nDone.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()