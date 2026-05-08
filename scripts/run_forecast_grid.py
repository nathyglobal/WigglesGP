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
    total_diagonal_sigma,
    gaussian_loglike_diagonal,
)

from wigglesgp.survey_forecast import (
    desi_like_survey,
    euclid_feature_paper_survey,
    roman_hlss_survey,
    survey_sigma_for_forecast_vector,
    combine_independent_sigmas,
    print_survey_summary,
)


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "fid_omega": 1.26,
        "omega_min": 1.20,
        "omega_max": 1.32,
        "n_omega": 71,
        "phi_min": 2.441592653589793,
        "phi_max": 3.841592653589793,
        "n_phi": 71,
        "output": Path("forecast_tests/log_forecast_grid_local_getdist_Afixed_71x71.csv"),
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "fid_omega": 0.87,
        "omega_min": 0.75,
        "omega_max": 0.99,
        "n_omega": 71,
        "phi_min": 2.441592653589793,
        "phi_max": 3.841592653589793,
        "n_phi": 71,
        "output": Path("forecast_tests/linear_forecast_grid_local_getdist_Afixed_71x71.csv"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a brute-force forecast likelihood grid over "
            "(A_feat, omega_label, phi) using the CAMB + Sigma-emulator model."
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
        "--fid-omega",
        type=float,
        default=None,
        help=(
            "Fiducial feature frequency label in the emulator convention. "
            "Defaults from --feature-type."
        ),
    )

    parser.add_argument(
        "--fid-phi",
        type=float,
        default=np.pi,
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
        "--omega-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit grid values for the emulator frequency label.",
    )

    parser.add_argument(
        "--omega-min",
        type=float,
        default=None,
        help="Minimum omega label if --omega-values is not supplied.",
    )

    parser.add_argument(
        "--omega-max",
        type=float,
        default=None,
        help="Maximum omega label if --omega-values is not supplied.",
    )

    parser.add_argument(
        "--n-omega",
        type=int,
        default=None,
        help="Number of omega-label grid points if explicit values are not supplied.",
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
        default=None,
        help="Minimum phi if --phi-values is not supplied.",
    )

    parser.add_argument(
        "--phi-max",
        type=float,
        default=None,
        help="Maximum phi if --phi-values is not supplied.",
    )

    parser.add_argument(
        "--n-phi",
        type=int,
        default=None,
        help="Number of phi grid points if explicit values are not supplied.",
    )

    parser.add_argument(
        "--frac-error",
        type=float,
        default=0.01,
        help="Simple fractional error assigned to each data-vector element.",
    )

    parser.add_argument(
        "--survey",
        choices=["none", "desi", "desi-extended", "euclid", "roman", "all"],
        default="none",
        help="Use a survey P(k,z) covariance instead of the toy fractional error.",
    )

    parser.add_argument(
        "--include-gp-uncertainty",
        action="store_true",
        help="Add propagated Sigma-emulator uncertainty to the likelihood covariance.",
    )

    parser.add_argument(
        "--model-error-floor",
        type=float,
        default=0.0,
        help=(
            "Additive model-error floor for the forecast observable. "
            "For ratio_nl, use e.g. 5e-3."
        ),
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
        default=None,
        help="Output CSV path. Defaults from --feature-type.",
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
    fid_omega = args.fid_omega if args.fid_omega is not None else defaults["fid_omega"]

    omega_min = args.omega_min if args.omega_min is not None else defaults["omega_min"]
    omega_max = args.omega_max if args.omega_max is not None else defaults["omega_max"]
    n_omega = args.n_omega if args.n_omega is not None else defaults["n_omega"]

    phi_min = args.phi_min if args.phi_min is not None else defaults["phi_min"]
    phi_max = args.phi_max if args.phi_max is not None else defaults["phi_max"]
    n_phi = args.n_phi if args.n_phi is not None else defaults["n_phi"]

    output = args.output or defaults["output"]

    return {
        "emulator_path": emulator_path,
        "fid_omega": fid_omega,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "n_omega": n_omega,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "n_phi": n_phi,
        "output": output,
    }


def build_survey_sigma(forecast, survey_name):
    if survey_name == "none":
        return None

    if survey_name == "desi":
        survey = desi_like_survey(extended=False)
        print_survey_summary(survey)
        return survey_sigma_for_forecast_vector(forecast, survey=survey)

    if survey_name == "desi-extended":
        survey = desi_like_survey(extended=True)
        print_survey_summary(survey)
        return survey_sigma_for_forecast_vector(forecast, survey=survey)

    if survey_name == "euclid":
        survey = euclid_feature_paper_survey()
        print_survey_summary(survey)
        return survey_sigma_for_forecast_vector(forecast, survey=survey)

    if survey_name == "roman":
        survey = roman_hlss_survey()
        print_survey_summary(survey)
        return survey_sigma_for_forecast_vector(forecast, survey=survey)

    if survey_name == "all":
        surveys = [
            desi_like_survey(extended=False),
            euclid_spectroscopic_survey(),
            roman_hlss_survey(),
        ]

        sigmas = []
        for survey in surveys:
            print_survey_summary(survey)
            sigmas.append(survey_sigma_for_forecast_vector(forecast, survey=survey))

        return combine_independent_sigmas(*sigmas)

    raise ValueError(f"Unknown survey_name={survey_name!r}.")


def build_grid_values(args, resolved):
    A_values = np.asarray(args.A_values, dtype=float)

    if args.omega_values is None:
        omega_values = np.linspace(
            resolved["omega_min"],
            resolved["omega_max"],
            resolved["n_omega"],
        )
    else:
        omega_values = np.asarray(args.omega_values, dtype=float)

    if args.phi_values is None:
        phi_values = np.linspace(
            resolved["phi_min"],
            resolved["phi_max"],
            resolved["n_phi"],
        )
    else:
        phi_values = np.asarray(args.phi_values, dtype=float)

    return A_values, omega_values, phi_values


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep log10omega column name for compatibility with existing GetDist plotter.
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
    resolved = resolve_defaults(args)
    check_domain = not args.no_domain_check

    A_values, omega_values, phi_values = build_grid_values(args, resolved)
    n_grid = len(A_values) * len(omega_values) * len(phi_values)

    emulator_path = resolved["emulator_path"]
    fid_omega = resolved["fid_omega"]
    output_path = resolved["output"]

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    print("\nForecast likelihood grid")
    print("------------------------")
    print(f"emulator:          {emulator_path}")
    print(f"feature_type:      {args.feature_type}")
    print(f"redshifts:         {args.redshifts}")
    print(f"observable:        {args.observable}")
    print(f"fid A_feat:        {args.fid_A_feat}")
    print(f"fid omega_label:   {fid_omega}")
    print(f"fid omega_model:   {10.0 ** fid_omega:.6g}")
    print(f"fid phi:           {args.fid_phi}")
    print(f"survey:            {args.survey}")
    print(f"frac_error:        {args.frac_error}")
    print(f"A grid:            {A_values}")
    print(f"omega grid:        {omega_values}")
    print(f"phi grid:          {phi_values}")
    print(f"n grid points:     {n_grid}")
    print(f"output:            {output_path}")

    emulator = SigmaEmulator.from_file(emulator_path)

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
        log10omega=fid_omega,
        A_feat=args.fid_A_feat,
        phi=args.fid_phi,
        feature_type=args.feature_type,
        kmax=args.kmax,
        npoints=args.npoints,
        k_fit_min=args.k_fit_min,
        k_fit_max=args.k_fit_max,
        check_domain=check_domain,
        observable=args.observable,
        propagate_gp_uncertainty=args.include_gp_uncertainty,
    )

    data = fid["vector"]
    survey_sigma = build_survey_sigma(fid, args.survey)

    sigma = total_diagonal_sigma(
        data,
        observational_sigma=survey_sigma,
        frac_error=args.frac_error if survey_sigma is None else None,
        gp_sigma=fid["gp_std"] if args.include_gp_uncertainty else None,
        model_error_floor=args.model_error_floor,
    )

    print(f"data-vector length: {len(data)}")
    print(f"k range:            {np.nanmin(fid['k']):.6g} -> {np.nanmax(fid['k']):.6g}")
    print(f"z range:            {np.nanmin(fid['z']):.6g} -> {np.nanmax(fid['z']):.6g}")

    print("\nUncertainty budget")
    print("------------------")
    if survey_sigma is None:
        print("observational covariance: toy fractional error")
        print(f"frac_error:               {args.frac_error}")
    else:
        print(f"observational covariance: survey {args.survey}")
        print("toy frac_error:           ignored")
        print(
            "survey sigma min/med/max: "
            f"{np.nanmin(survey_sigma):.6g} / "
            f"{np.nanmedian(survey_sigma):.6g} / "
            f"{np.nanmax(survey_sigma):.6g}"
        )

    print(f"include GP:               {args.include_gp_uncertainty}")

    if args.include_gp_uncertainty:
        gp = fid["gp_std"]
        print(
            "GP std min/med/max:       "
            f"{np.nanmin(gp):.6g} / "
            f"{np.nanmedian(gp):.6g} / "
            f"{np.nanmax(gp):.6g}"
        )

    print(f"model_error_floor:        {args.model_error_floor}")
    print(
        "total sigma min/med/max:  "
        f"{np.nanmin(sigma):.6g} / "
        f"{np.nanmedian(sigma):.6g} / "
        f"{np.nanmax(sigma):.6g}"
    )

    rows = []

    print("\nEvaluating grid")
    print("---------------")

    t_start = time.time()

    for counter, (A_feat, omega_label, phi) in enumerate(
        itertools.product(A_values, omega_values, phi_values),
        start=1,
    ):
        t0 = time.time()

        model = emulator_damped_camb_data_vector(
            emulator=emulator,
            vanilla_cache=vanilla_cache,
            redshifts=args.redshifts,
            log10omega=omega_label,
            A_feat=A_feat,
            phi=phi,
            feature_type=args.feature_type,
            kmax=args.kmax,
            npoints=args.npoints,
            k_fit_min=args.k_fit_min,
            k_fit_max=args.k_fit_max,
            check_domain=check_domain,
            observable=args.observable,
            propagate_gp_uncertainty=args.include_gp_uncertainty,
        )

        loglike, chi2 = gaussian_loglike_diagonal(
            model["vector"],
            data,
            sigma,
        )

        runtime = time.time() - t0

        row = {
            "A_feat": float(A_feat),
            "log10omega": float(omega_label),
            "phi": float(phi),
            "loglike": float(loglike),
            "chi2": float(chi2),
            "delta_chi2": float(chi2),
            "n_data": float(len(data)),
            "runtime_s": float(runtime),
        }

        rows.append(row)

        print(
            f"[{counter:4d}/{n_grid}] "
            f"A={A_feat:.5g}, "
            f"omega_label={omega_label:.5g}, "
            f"phi={phi:.5g}, "
            f"chi2={chi2:.6g}, "
            f"runtime={runtime:.2f}s"
        )

    write_rows(output_path, rows)

    print("\nDone.")
    print(f"Total runtime: {time.time() - t_start:.2f}s")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()