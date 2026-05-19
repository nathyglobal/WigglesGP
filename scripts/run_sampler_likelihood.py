#!/usr/bin/env python3

from pathlib import Path
import argparse
import time
import numpy as np

from wigglesgp.likelihood import WiggleComparisonConfig, WiggleComparisonLikelihood


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "fid_omega": 1.26,
        "omega_min": 0.8,
        "omega_max": 2.0,
        "output_prefix": "log_comparison",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BOBE likelihood comparison for logarithmic primordial features."
    )

    parser.add_argument(
        "--scenario",
        choices=["full_damped", "full_undamped", "linear_only"],
        required=True,
        help="Comparison scenario to run.",
    )

    parser.add_argument(
        "--emulator",
        type=Path,
        default=None,
        help="Path to saved Sigma emulator.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path.",
    )

    parser.add_argument(
        "--fid-A-feat",
        type=float,
        default=0.03,
        help="Injected feature amplitude.",
    )

    parser.add_argument(
        "--fid-omega",
        type=float,
        default=None,
        help="Injected log10 omega.",
    )

    parser.add_argument(
        "--fid-phi",
        type=float,
        default=np.pi,
        help="Injected phase in radians.",
    )

    parser.add_argument(
        "--A-min",
        type=float,
        default=0.0,
        help="Minimum feature-amplitude prior.",
    )

    parser.add_argument(
        "--A-max",
        type=float,
        default=0.06,
        help="Maximum feature-amplitude prior.",
    )

    parser.add_argument(
        "--omega-min",
        type=float,
        default=None,
        help="Minimum log10 omega prior.",
    )

    parser.add_argument(
        "--omega-max",
        type=float,
        default=None,
        help="Maximum log10 omega prior.",
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
        "--model-error-floor",
        type=float,
        default=5e-3,
        help="Additive model-error floor on the ratio observable.",
    )

    parser.add_argument(
        "--include-gp-uncertainty",
        action="store_true",
        help="Include propagated GP uncertainty in the likelihood covariance.",
    )

    parser.add_argument(
        "--internal-npoints",
        type=int,
        default=2000,
        help="Internal k-grid size before top-hat binning.",
    )

    parser.add_argument(
        "--bin-delta-k",
        type=float,
        default=0.004,
        help="Top-hat k-bin width in h/Mpc. Set <=0 to disable binning.",
    )

    parser.add_argument(
        "--k-fit-max",
        type=float,
        default=0.6,
        help="Maximum k-bin edge",
    )

    parser.add_argument(
        "--k-fit-min",
        type=float,
        default=0.05,
        help="Minimum k-bin edge",
    )

    parser.add_argument(
        "--keep-partial-final-bin",
        action="store_true",
        help="Keep the final partial k-bin instead of dropping it.",
    )

    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.05,
        help="BOBE stopping threshold.",
    )

    parser.add_argument(
        "--max-evals",
        type=int,
        default=500,
        help="Maximum BOBE likelihood evaluations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    defaults = DEFAULTS["log"]

    emulator_path = args.emulator or defaults["emulator"]
    fid_omega = args.fid_omega if args.fid_omega is not None else defaults["fid_omega"]
    omega_min = args.omega_min if args.omega_min is not None else defaults["omega_min"]
    omega_max = args.omega_max if args.omega_max is not None else defaults["omega_max"]

    if args.output is None:
        output = Path("comparison_tests") / f"{defaults['output_prefix']}_{args.scenario}.npz"
    else:
        output = args.output

    config = WiggleComparisonConfig(
        feature_type="log",
        emulator_path=emulator_path,
        scenario=args.scenario,
        observable="power",
        fid_A_feat=args.fid_A_feat,
        fid_omega=fid_omega,
        fid_phi=args.fid_phi,
        A_min=args.A_min,
        A_max=args.A_max,
        omega_min=omega_min,
        omega_max=omega_max,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        internal_npoints=args.internal_npoints,
        k_fit_max=args.k_fit_max,
        k_fit_min=args.k_fit_min,
        bin_delta_k=args.bin_delta_k,
        keep_partial_final_bin=args.keep_partial_final_bin,
        include_gp_uncertainty=args.include_gp_uncertainty,
        model_error_floor=args.model_error_floor,
        check_domain=not args.no_domain_check,
    )

    likelihood = WiggleComparisonLikelihood(config)
    print(likelihood.summary())

    theta_fid = np.array([config.fid_A_feat, config.fid_omega, config.fid_phi])
    print(f"\nchi2(fid): {likelihood.chi2(theta_fid):.6g}")

    t0 = time.time()

    def loglike(theta):
        ll = likelihood.loglike(theta)

        if likelihood.eval_counter % 100 == 0:
            elapsed = time.time() - t0
            chi2 = likelihood.chi2(theta)
            A_feat, omega_label, phi = theta
            print(
                f"eval {likelihood.eval_counter:6d}: "
                f"A={A_feat:.6g}, omega={omega_label:.6g}, phi={phi:.6g}, "
                f"chi2={chi2:.6g}, elapsed={elapsed:.1f}s"
            )

        return ll

    from BOBE import BOBE

    likelihood_name = f"log_comparison_{args.scenario}"

    bobe = BOBE(
        loglikelihood=loglike,
        param_list=likelihood.param_list,
        param_bounds=likelihood.param_bounds,
        param_labels=likelihood.param_labels,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        resume=False,
        resume_file=f"./comparison_tests/{likelihood_name}",
        save_dir="./comparison_tests/",
        save=True,
        verbosity="INFO",
        n_sobol_init=12,
        use_clf=False,
        minus_inf=-1e5,
        seed=args.seed,
    )

    results = bobe.run(
        acq="wipstd",
        min_evals=1,
        max_evals=args.max_evals,
        max_gp_size=args.max_evals,
        fit_n_points=2,
        ns_n_points=2,
        batch_size=2,
        num_hmc_warmup=128,
        num_hmc_samples=512,
        mc_points_size=256,
        logz_threshold=args.dlogz,
        do_final_ns=False,
    )

    if results is None:
        raise RuntimeError("BOBE returned no results.")

    samples = results["samples"]
    sample_array = samples["x"]
    weights_array = samples["weights"]

    # logz_dict = results.get("logz", {})
    # logz = logz_dict.get("mean", np.nan)
    # logzerr = 0.5 * (logz_dict["upper"] - logz_dict["lower"])

    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output,
        samples=sample_array,
        weights=weights_array,
        #logz=logz,
        #logzerr=logzerr,
        parameter_names=np.array(likelihood.param_list),
        feature_type="log",
        scenario=args.scenario,
        observable=config.observable,
        fid_A_feat=config.fid_A_feat,
        fid_omega=config.fid_omega,
        fid_phi=config.fid_phi,
        A_min=config.A_min,
        A_max=config.A_max,
        omega_min=config.omega_min,
        omega_max=config.omega_max,
        phi_min=config.phi_min,
        phi_max=config.phi_max,
        redshifts=np.asarray(config.redshifts, dtype=float),
        model_error_floor=config.model_error_floor,
        include_gp_uncertainty=config.include_gp_uncertainty,
        apply_damping=likelihood.apply_damping,
        bin_delta_k=config.bin_delta_k,
        internal_npoints=config.internal_npoints,
        keep_partial_final_bin=config.keep_partial_final_bin,
        is_binned=bool(config.bin_delta_k > 0.0),
        linear_kmax_by_z=np.array(
            sorted(likelihood.k_fit_max_by_z.items())
            if likelihood.k_fit_max_by_z is not None else [],
            dtype=float,
        ),
    )

    print("\nDone.")
    print(f"n likelihood evaluations: {likelihood.eval_counter}")
    #print(f"logZ: {logz:.6g} +/- {logzerr:.6g}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()