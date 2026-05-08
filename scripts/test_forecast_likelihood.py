#!/usr/bin/env python3

from pathlib import Path
import argparse

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
    euclid_spectroscopic_survey,
    roman_hlss_survey,
    survey_sigma_for_forecast_vector,
    combine_independent_sigmas,
    print_survey_summary,
)


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "fid_omega": 1.26,
        "output": Path("forecast_tests/log_forecast_likelihood_scan.csv"),
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "fid_omega": 0.87,
        "output": Path("forecast_tests/linear_forecast_likelihood_scan.csv"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple CAMB + Sigma-emulator forecast likelihood "
            "for primordial-feature parameters."
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
        "--fid-omega",
        type=float,
        default=None,
        help=(
            "Fiducial feature frequency label in the emulator convention. "
            "Defaults from --feature-type."
        ),
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
        default=None,
        help="Output CSV containing test likelihood evaluations. Defaults from --feature-type.",
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
    output = args.output or defaults["output"]

    return emulator_path, fid_omega, output


def build_test_points(fid_A, fid_omega, fid_phi):
    """
    Small deterministic set of likelihood test points.

    This is not a sampler. It is just a redline test that the likelihood
    responds sensibly to A, frequency, and phase.
    """
    return [
        {
            "label": "fiducial",
            "A_feat": fid_A,
            "omega_label": fid_omega,
            "phi": fid_phi,
        },
        {
            "label": "lower_A",
            "A_feat": 0.8 * fid_A,
            "omega_label": fid_omega,
            "phi": fid_phi,
        },
        {
            "label": "higher_A",
            "A_feat": 1.2 * fid_A,
            "omega_label": fid_omega,
            "phi": fid_phi,
        },
        {
            "label": "lower_omega",
            "A_feat": fid_A,
            "omega_label": fid_omega - 0.05,
            "phi": fid_phi,
        },
        {
            "label": "higher_omega",
            "A_feat": fid_A,
            "omega_label": fid_omega + 0.05,
            "phi": fid_phi,
        },
        {
            "label": "phase_plus_0p25",
            "A_feat": fid_A,
            "omega_label": fid_omega,
            "phi": fid_phi + 0.25,
        },
        {
            "label": "phase_plus_0p5",
            "A_feat": fid_A,
            "omega_label": fid_omega,
            "phi": fid_phi + 0.5,
        },
    ]


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
        survey = euclid_spectroscopic_survey()
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


def write_results(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "label",
        "A_feat",
        "omega_label",
        "omega_model",
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
    emulator_path, fid_omega, output_path = resolve_defaults(args)
    check_domain = not args.no_domain_check

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    emulator = SigmaEmulator.from_file(emulator_path)

    print("\nBuilding fiducial forecast data vector")
    print("--------------------------------------")
    print(f"emulator:       {emulator_path}")
    print(f"feature_type:   {args.feature_type}")
    print(f"redshifts:      {args.redshifts}")
    print(f"observable:     {args.observable}")
    print(f"fid A_feat:     {args.fid_A_feat}")
    print(f"fid omega_label:{fid_omega}")
    print(f"fid omega_model:{10.0 ** fid_omega:.6g}")
    print(f"fid phi:        {args.fid_phi}")
    print(f"survey:         {args.survey}")
    print(f"frac_error:     {args.frac_error}")

    print("\nBuilding vanilla CAMB spectra cache")
    print("----------------------------------")

    vanilla_cache = build_vanilla_spectra_cache(
        redshifts=args.redshifts,
        kmax=args.kmax,
        npoints=args.npoints,
    )

    print(f"cached redshifts: {sorted(vanilla_cache.keys())}")

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
        print(f"observational covariance: toy fractional error")
        print(f"frac_error:               {args.frac_error}")
    else:
        print(f"observational covariance: survey {args.survey}")
        print(f"toy frac_error:           ignored")
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

    test_points = build_test_points(
        args.fid_A_feat,
        fid_omega,
        args.fid_phi,
    )

    rows = []

    print("\nLikelihood test points")
    print("----------------------")

    fid_chi2 = None

    for point in test_points:
        model = emulator_damped_camb_data_vector(
            emulator=emulator,
            vanilla_cache=vanilla_cache,
            redshifts=args.redshifts,
            log10omega=point["omega_label"],
            A_feat=point["A_feat"],
            phi=point["phi"],
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

        if point["label"] == "fiducial":
            fid_chi2 = chi2

        delta_chi2 = chi2 - fid_chi2 if fid_chi2 is not None else np.nan

        row = {
            "label": point["label"],
            "A_feat": point["A_feat"],
            "omega_label": point["omega_label"],
            "omega_model": 10.0 ** point["omega_label"],
            "phi": point["phi"],
            "loglike": loglike,
            "chi2": chi2,
            "delta_chi2": delta_chi2,
        }

        rows.append(row)

        print(
            f"{point['label']:>16s}: "
            f"A={point['A_feat']:.5g}, "
            f"omega_label={point['omega_label']:.5g}, "
            f"phi={point['phi']:.5g}, "
            f"chi2={chi2:.6g}, "
            f"Delta chi2={delta_chi2:.6g}"
        )

    write_results(output_path, rows)

    print("\nDone.")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()