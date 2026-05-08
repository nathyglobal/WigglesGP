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
    euclid_feature_paper_survey,
    survey_sigma_for_forecast_vector,
    print_survey_summary,
)


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "fid_omega": 1.26,
        "output": Path("forecast_tests/log_no_feature_comparison_euclid.csv"),
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "fid_omega": 0.87,
        "output": Path("forecast_tests/linear_no_feature_comparison_euclid.csv"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the no-feature Delta chi2 for a fiducial primordial-feature "
            "forecast data vector."
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
        help="Saved Sigma emulator. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--redshifts",
        type=float,
        nargs="+",
        default=[1.0, 1.2, 1.4, 1.65],
        help="Euclid GCsp redshift bins.",
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
        help="Fiducial emulator frequency label. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--fid-phi",
        type=float,
        default=np.pi,
        help="Fiducial phase in radians.",
    )

    parser.add_argument(
        "--include-gp-uncertainty",
        action="store_true",
        help="Include propagated Sigma-emulator uncertainty in the covariance.",
    )

    parser.add_argument(
        "--model-error-floor",
        type=float,
        default=5e-3,
        help="Additive model-error floor for ratio_nl.",
    )

    parser.add_argument(
        "--kmax",
        type=float,
        default=0.8,
        help="Maximum CAMB k/h.",
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
        help="Minimum fitted k/h.",
    )

    parser.add_argument(
        "--k-fit-max",
        type=float,
        default=0.6,
        help="Maximum fitted k/h.",
    )

    parser.add_argument(
        "--observable",
        choices=["ratio_nl"],
        default="ratio_nl",
        help="Currently only ratio_nl is supported for the analytic no-feature comparison.",
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

    return {
        "emulator": args.emulator or defaults["emulator"],
        "fid_omega": args.fid_omega if args.fid_omega is not None else defaults["fid_omega"],
        "output": args.output or defaults["output"],
    }


def write_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "feature_type",
        "fid_A_feat",
        "fid_omega",
        "fid_phi",
        "n_data",
        "chi2_no_feature",
        "snr_no_feature",
        "min_data",
        "median_data",
        "max_data",
        "min_sigma",
        "median_sigma",
        "max_sigma",
    ]

    with open(path, "w") as handle:
        handle.write(",".join(columns) + "\n")
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
    resolved = resolve_defaults(args)
    check_domain = not args.no_domain_check

    emulator_path = resolved["emulator"]
    fid_omega = resolved["fid_omega"]
    output = resolved["output"]

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    emulator = SigmaEmulator.from_file(emulator_path)

    print("\nNo-feature comparison")
    print("---------------------")
    print(f"feature_type:       {args.feature_type}")
    print(f"emulator:           {emulator_path}")
    print(f"redshifts:          {args.redshifts}")
    print(f"fid A_feat:         {args.fid_A_feat}")
    print(f"fid omega_label:    {fid_omega}")
    print(f"fid omega_model:    {10.0 ** fid_omega:.6g}")
    print(f"fid phi:            {args.fid_phi}")
    print(f"include GP:         {args.include_gp_uncertainty}")
    print(f"model_error_floor:  {args.model_error_floor}")

    print("\nBuilding vanilla CAMB spectra cache")
    print("----------------------------------")

    vanilla_cache = build_vanilla_spectra_cache(
        redshifts=args.redshifts,
        kmax=args.kmax,
        npoints=args.npoints,
    )

    print(f"cached redshifts: {sorted(vanilla_cache.keys())}")

    print("\nBuilding fiducial feature data vector")
    print("-------------------------------------")

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

    survey = euclid_feature_paper_survey()
    print_survey_summary(survey)

    survey_sigma = survey_sigma_for_forecast_vector(fid, survey=survey)

    sigma = total_diagonal_sigma(
        data,
        observational_sigma=survey_sigma,
        frac_error=None,
        gp_sigma=fid["gp_std"] if args.include_gp_uncertainty else None,
        model_error_floor=args.model_error_floor,
    )

    no_feature = np.ones_like(data)

    loglike_no_feature, chi2_no_feature = gaussian_loglike_diagonal(
        no_feature,
        data,
        sigma,
    )

    snr_no_feature = np.sqrt(chi2_no_feature)

    print("\nData-vector summary")
    print("-------------------")
    print(f"n_data:             {len(data)}")
    print(f"k range:            {np.nanmin(fid['k']):.6g} -> {np.nanmax(fid['k']):.6g}")
    print(f"z range:            {np.nanmin(fid['z']):.6g} -> {np.nanmax(fid['z']):.6g}")
    print(
        "data min/med/max:   "
        f"{np.nanmin(data):.6g} / {np.nanmedian(data):.6g} / {np.nanmax(data):.6g}"
    )
    print(
        "sigma min/med/max:  "
        f"{np.nanmin(sigma):.6g} / {np.nanmedian(sigma):.6g} / {np.nanmax(sigma):.6g}"
    )

    print("\nNo-feature result")
    print("-----------------")
    print(f"chi2 no feature:    {chi2_no_feature:.6g}")
    print(f"Delta chi2:         {chi2_no_feature:.6g}")
    print(f"approx S/N:         {snr_no_feature:.6g}")
    print(f"loglike no feature: {loglike_no_feature:.6g}")

    row = {
        "feature_type": args.feature_type,
        "fid_A_feat": args.fid_A_feat,
        "fid_omega": fid_omega,
        "fid_phi": args.fid_phi,
        "n_data": len(data),
        "chi2_no_feature": chi2_no_feature,
        "snr_no_feature": snr_no_feature,
        "min_data": np.nanmin(data),
        "median_data": np.nanmedian(data),
        "max_data": np.nanmax(data),
        "min_sigma": np.nanmin(sigma),
        "median_sigma": np.nanmedian(sigma),
        "max_sigma": np.nanmax(sigma),
    }

    write_result(output, row)

    print("\nDone.")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()