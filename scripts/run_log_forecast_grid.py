#!/usr/bin/env python3

from pathlib import Path
import argparse
import itertools
import time

import numpy as np

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.camb_power import build_vanilla_spectra_cache
from wigglesgp.forecasting import (
    emulator_damped_camb_data_vector,
    diagonal_fractional_sigma,
    gaussian_loglike_diagonal,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a brute-force forecast likelihood grid over "
            "(A_feat, log10omega, phi) using the CAMB + Sigma-emulator model."
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
        "--fid-A-feat",
        type=float,
        default=0.03,
        help="Fiducial feature amplitude.",
    )

    parser.add_argument(
        "--fid-log10omega",
        type=float,
        default=1.26,
        help="Fiducial log10 feature frequency.",
    )

    parser.add_argument(
        "--fid-phi",
        type=float,
        default=0.0,
        help="Fiducial feature phase in radians.",
    )

    parser.add_argument(
        "--A-values",
        type=float,
        nargs="+",
        default=[0.03],
        help="Grid values for A_feat.",
    )

    parser.add_argument(
        "--log10omega-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit grid values for log10omega.",
    )

    parser.add_argument(
        "--log10omega-min",
        type=float,
        default=1.16,
        help="Minimum log10omega if --log10omega-values is not supplied.",
    )

    parser.add_argument(
        "--log10omega-max",
        type=float,
        default=1.36,
        help="Maximum log10omega if --log10omega-values is not supplied.",
    )

    parser.add_argument(
        "--n-log10omega",
        type=int,
        default=9,
        help="Number of log10omega grid points if explicit values are not supplied.",
    )

    parser.add_argument(
        "--phi-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit grid values for phi.",
    )

    parser.add_argument(
        "--phi-min",
        type=float,
        default=-0.5,
        help="Minimum phi if --phi-values is not supplied.",
    )

    parser.add_argument(
        "--phi-max",
        type=float,
        default=0.5,
        help="Maximum phi if --phi-values is not supplied.",
    )

    parser.add_argument(
        "--n-phi",
        type=int,
        default=9,
        help="Number of phi grid points if explicit values are not supplied.",
    )

    parser.add_argument(
        "--frac-error",
        type=float,
        default=0.01,
        help="Simple fractional error assigned to each data-vector element.",
    )

    parser.add_argument(
        "--observable",
        choices=["ratio_nl", "p_wig_nl"],
        default="ratio_nl",
        help="Forecast observable.",
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
        "--output",
        type=Path,
        default=Path("forecast_tests/log_forecast_grid.csv"),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def build_grid_values(args):
    A_values = np.asarray(args.A_values, dtype=float)

    if args.log10omega_values is None:
        log10omega_values = np.linspace(
            args.log10omega_min,
            args.log10omega_max,
            args.n_log10omega,
        )
    else:
        log10omega_values = np.asarray(args.log10omega_values, dtype=float)

    if args.phi_values is None:
        phi_values = np.linspace(args.phi_min, args.phi_max, args.n_phi)
    else:
        phi_values = np.asarray(args.phi_values, dtype=float)

    return A_values, log10omega_values, phi_values


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "A_feat",
        "log10omega",
        "phi",
        "loglike",
        "chi2",
        "delta_chi2",
        "n_data",
        "runtime_s",
    ]

    with open(path, "w") as handle:
        handle.write(",".join(columns) + "\n")

        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                values.append(f"{value:.12g}")
            handle.write(",".join(values) + "\n")


def main():
    args = parse_args()
    check_domain = not args.no_domain_check

    A_values, log10omega_values, phi_values = build_grid_values(args)

    n_grid = len(A_values) * len(log10omega_values) * len(phi_values)

    print("\nForecast likelihood grid")
    print("------------------------")
    print(f"emulator:          {args.emulator}")
    print(f"feature_type:      {args.feature_type}")
    print(f"redshifts:         {args.redshifts}")
    print(f"observable:        {args.observable}")
    print(f"fid A_feat:        {args.fid_A_feat}")
    print(f"fid log10omega:    {args.fid_log10omega}")
    print(f"fid phi:           {args.fid_phi}")
    print(f"frac_error:        {args.frac_error}")
    print(f"A grid:            {A_values}")
    print(f"log10omega grid:   {log10omega_values}")
    print(f"phi grid:          {phi_values}")
    print(f"n grid points:     {n_grid}")
    print(f"output:            {args.output}")

    emulator = SigmaEmulator.from_file(args.emulator)


    print("\nBuilding vanilla CAMB spectra cache")
    print("----------------------------------")

    vanilla_cache = build_vanilla_spectra_cache(
        redshifts=args.redshifts,
        kmax=args.kmax,
        npoints=args.npoints,
    )

    print(f"cached redshifts: {sorted(vanilla_cache.keys())}")

    print("\nBuilding fiducial data vector")
    print("-----------------------------")

    fid = emulator_damped_camb_data_vector(
        emulator=emulator,
        vanilla_cache=vanilla_cache,
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

    rows = []

    print("\nEvaluating grid")
    print("---------------")

    t_start = time.time()

    for counter, (A_feat, log10omega, phi) in enumerate(
        itertools.product(A_values, log10omega_values, phi_values),
        start=1,
    ):
        t0 = time.time()

        model = emulator_damped_camb_data_vector(
            emulator=emulator,
            vanilla_cache=vanilla_cache,
            redshifts=args.redshifts,
            log10omega=log10omega,
            A_feat=A_feat,
            phi=phi,
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

        runtime = time.time() - t0

        row = {
            "A_feat": float(A_feat),
            "log10omega": float(log10omega),
            "phi": float(phi),
            "loglike": float(loglike),
            "chi2": float(chi2),
            "delta_chi2": float(chi2),  # fiducial chi2 is zero by construction
            "n_data": float(len(data)),
            "runtime_s": float(runtime),
        }

        rows.append(row)

        print(
            f"[{counter:4d}/{n_grid}] "
            f"A={A_feat:.5g}, "
            f"log10omega={log10omega:.5g}, "
            f"phi={phi:.5g}, "
            f"chi2={chi2:.6g}, "
            f"runtime={runtime:.2f}s"
        )

    write_rows(args.output, rows)

    print("\nDone.")
    print(f"Total runtime: {time.time() - t_start:.2f}s")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()