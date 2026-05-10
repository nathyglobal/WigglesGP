#!/usr/bin/env python3

from pathlib import Path
import argparse
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
    euclid_feature_paper_survey,
    survey_sigma_for_forecast_vector,
    print_survey_summary,
)


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "fid_omega": 1.26,
        "omega_min": 1.20,
        "omega_max": 1.32,
        "output": Path("forecast_tests/BOBE_log_forecast_sampler_euclid_2d.npz"),
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "fid_omega": 0.87,
        "omega_min": 0.75,
        "omega_max": 0.99,
        "output": Path("forecast_tests/BOBE_linear_forecast_sampler_euclid_2d.npz"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a nested-sampling forecast over omega_label and phi."
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
        help="Survey redshifts.",
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
        "--omega-min",
        type=float,
        default=None,
        help="Minimum prior omega label. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--omega-max",
        type=float,
        default=None,
        help="Maximum prior omega label. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--phi-min",
        type=float,
        default=0.0,
        help="Minimum phase prior.",
    )

    parser.add_argument(
        "--phi-max",
        type=float,
        default=2.0 * np.pi,
        help="Maximum phase prior.",
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
        help="Number of CAMB k points.",
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
        choices=["ratio_nl", "p_wig_nl"],
        default="ratio_nl",
        help="Forecast observable.",
    )

    # parser.add_argument(
    #     "--nlive",
    #     type=int,
    #     default=250,
    #     help="Number of live points for dynesty.",
    # )

    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.05,
        help="BOBE convergence criterion.",
    )

    # parser.add_argument(
    #     "--sample",
    #     choices=["auto", "rwalk", "rslice", "slice"],
    #     default="rwalk",
    #     help="Dynesty sampling method.",
    # )

    # parser.add_argument(
    #     "--bound",
    #     choices=["multi", "single", "balls", "cubes", "none"],
    #     default="multi",
    #     help="Dynesty bounding method.",
    # )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    d = DEFAULTS[args.feature_type]

    return {
        "emulator": args.emulator or d["emulator"],
        "fid_omega": args.fid_omega if args.fid_omega is not None else d["fid_omega"],
        "omega_min": args.omega_min if args.omega_min is not None else d["omega_min"],
        "omega_max": args.omega_max if args.omega_max is not None else d["omega_max"],
        "output": args.output or d["output"],
    }


def main():
    args = parse_args()
    resolved = resolve_defaults(args)
    check_domain = not args.no_domain_check

    #import dynesty

    emulator_path = resolved["emulator"]
    fid_omega = resolved["fid_omega"]
    omega_min = resolved["omega_min"]
    omega_max = resolved["omega_max"]
    output = resolved["output"]

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    emulator = SigmaEmulator.from_file(emulator_path)

    print("\nForecast sampler")
    print("----------------")
    print(f"feature_type:      {args.feature_type}")
    print(f"emulator:          {emulator_path}")
    print(f"redshifts:         {args.redshifts}")
    print(f"fid A_feat:        {args.fid_A_feat}")
    print(f"fid omega_label:   {fid_omega}")
    print(f"fid phi:           {args.fid_phi}")
    print(f"omega prior:       [{omega_min}, {omega_max}]")
    print(f"phi prior:         [{args.phi_min}, {args.phi_max}]")
    print(f"nlive:             {args.nlive}")
    print(f"dlogz:             {args.dlogz}")
    print(f"output:            {output}")

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

    print("\nUncertainty budget")
    print("------------------")
    print(
        "survey sigma min/med/max: "
        f"{np.nanmin(survey_sigma):.6g} / "
        f"{np.nanmedian(survey_sigma):.6g} / "
        f"{np.nanmax(survey_sigma):.6g}"
    )
    if args.include_gp_uncertainty:
        gp = fid["gp_std"]
        print(
            "GP std min/med/max:     "
            f"{np.nanmin(gp):.6g} / "
            f"{np.nanmedian(gp):.6g} / "
            f"{np.nanmax(gp):.6g}"
        )
    print(f"model_error_floor:      {args.model_error_floor}")
    print(
        "total sigma min/med/max:"
        f" {np.nanmin(sigma):.6g} / "
        f"{np.nanmedian(sigma):.6g} / "
        f"{np.nanmax(sigma):.6g}"
    )

    ndim = 2
    param_bounds = np.array(
    [
        [omega_min, omega_max],
        [args.phi_min, args.phi_max],
    ],
    dtype=float,
    ).T
    param_list = ["omega_label", "phi"]

    param_labels = [
        r"\log_{10}(\omega)",
        r"\phi",
    ]

    # def prior_transform(u):
    #     u = np.asarray(u)
    #     omega = omega_min + u[0] * (omega_max - omega_min)
    #     phi = args.phi_min + u[1] * (args.phi_max - args.phi_min)
    #     return np.array([omega, phi])

    eval_counter = {"n": 0}
    t0 = time.time()

    def loglike(theta):
        omega_label, phi = theta
        eval_counter["n"] += 1

        model = emulator_damped_camb_data_vector(
            emulator=emulator,
            vanilla_cache=vanilla_cache,
            redshifts=args.redshifts,
            log10omega=omega_label,
            A_feat=args.fid_A_feat,
            phi=phi,
            feature_type=args.feature_type,
            kmax=args.kmax,
            npoints=args.npoints,
            k_fit_min=args.k_fit_min,
            k_fit_max=args.k_fit_max,
            check_domain=check_domain,
            observable=args.observable,
            propagate_gp_uncertainty=False,
        )

        ll, chi2 = gaussian_loglike_diagonal(
            model["vector"],
            data,
            sigma,
        )

        if eval_counter["n"] % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"eval {eval_counter['n']:6d}: "
                f"omega={omega_label:.6g}, phi={phi:.6g}, "
                f"chi2={chi2:.6g}, elapsed={elapsed:.1f}s"
            )

        return ll

    # sampler = dynesty.NestedSampler(
    #     loglike,
    #     prior_transform,
    #     ndim,
    #     nlive=args.nlive,
    #     bound=args.bound,
    #     sample=args.sample,
    # )

    # sampler.run_nested(dlogz=args.dlogz, print_progress=True)

    # results = sampler.results

    from BOBE import BOBE

    likelihood_name = f"forecast_{args.feature_type}_euclid_2d"
    bobe = BOBE(
            loglikelihood=loglike,
            param_list=param_list,
            param_bounds=param_bounds,
            param_labels=param_labels,
            likelihood_name=likelihood_name,
            confidence_for_unbounded=0.9999995,
            resume=False,
            resume_file=f'{likelihood_name}',
            save_dir='./forecast_tests/',
            save=True,
            verbosity='INFO',
            n_sobol_init=4,
            use_clf=False,
            minus_inf=-1e5,
            seed=12345,
            #gp_kwargs=gp_kwargs,
        )
            
    results = bobe.run(
        acq='wipstd',
        min_evals=1, 
        max_evals=250,
        max_gp_size=250,
        fit_n_points=2, 
        ns_n_points=2,
        batch_size=2,
        num_hmc_warmup=512,
        num_hmc_samples=1024, 
        mc_points_size=512,
        logz_threshold=args.dlogz,
        do_final_ns=True,
    )
    if results is not None:
        samples = results['samples']
        sample_array = samples['x']
        weights_array = samples['weights']
        logz_dict = results.get('logz', {})
        logz = logz_dict.get('mean', 'N/A')
        logzerr = (logz_dict['upper'] - logz_dict['lower'])/2

        output.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output,
            samples=sample_array,
            #logwt=weights_array,
            logz=logz,
            logzerr=logzerr,
            #logl=results.logl,
            weights=weights_array,
            parameter_names=np.array(["omega_label", "phi"]),
            feature_type=args.feature_type,
            fid_A_feat=args.fid_A_feat,
            fid_omega=fid_omega,
            fid_phi=args.fid_phi,
            omega_min=omega_min,
            omega_max=omega_max,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            redshifts=np.asarray(args.redshifts, dtype=float),
            model_error_floor=args.model_error_floor,
            include_gp_uncertainty=args.include_gp_uncertainty,
        )

    print("\nDone.")
    print(f"n likelihood evaluations: {eval_counter['n']}")
    print(f"logZ: {logz:.6g} +/- {logzerr:.6g}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()