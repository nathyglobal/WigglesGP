#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from wigglesgp.likelihood import WiggleComparisonConfig, WiggleComparisonLikelihood


SCENARIOS = ("full_damped", "full_undamped", "linear_only")


def make_config(
    *,
    observable="power",
    scenario="full_damped",
    feature_type="log",
    emulator_path=Path("emulators/log_sigma_gp.pkl"),
    fid_A_feat=0.03,
    fid_omega=1.26,
    fid_phi=np.pi,
    omega_min=0.8,
    omega_max=2.0,
    A_min=0.0,
    A_max=0.06,
    phi_min=0.0,
    phi_max=2.0 * np.pi,
    redshifts=(1.0, 1.2, 1.4, 1.65),
    kmax=0.8,
    k_fit_min=0.05,
    k_fit_max=0.6,
    internal_npoints=2000,
    bin_delta_k=0.004,
    keep_partial_final_bin=False,
    include_gp_uncertainty=True,
    model_error_floor=5e-3,
    check_domain=True,
):
    return WiggleComparisonConfig(
        feature_type=feature_type,
        emulator_path=Path(emulator_path),
        scenario=scenario,
        observable=observable,
        redshifts=tuple(float(z) for z in redshifts),
        fid_A_feat=float(fid_A_feat),
        fid_omega=float(fid_omega),
        fid_phi=float(fid_phi),
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        A_min=float(A_min),
        A_max=float(A_max),
        phi_min=float(phi_min),
        phi_max=float(phi_max),
        kmax=float(kmax),
        k_fit_min=float(k_fit_min),
        k_fit_max=float(k_fit_max),
        internal_npoints=int(internal_npoints),
        bin_delta_k=float(bin_delta_k),
        keep_partial_final_bin=bool(keep_partial_final_bin),
        include_gp_uncertainty=bool(include_gp_uncertainty),
        model_error_floor=float(model_error_floor),
        check_domain=bool(check_domain),
    )


def fiducial_theta(like):
    return np.array(
        [
            like.config.fid_A_feat,
            like.config.fid_omega,
            like.config.fid_phi,
        ],
        dtype=float,
    )


def default_theta_tests(like):
    return [
        fiducial_theta(like),
        np.array([0.025, 1.20, 2.8], dtype=float),
        np.array([0.035, 1.35, 3.4], dtype=float),
        np.array([0.015, 1.00, 1.0], dtype=float),
    ]


def print_header(title):
    print("\n" + title)
    print("=" * len(title))


def max_abs(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(np.abs(x)))


def concat_block_key(prediction, key):
    return np.concatenate(
        [np.asarray(block[key], dtype=float) for block in prediction["blocks"]]
    )


def test_fiducial_self_consistency(base_kwargs):
    """
    The fiducial Asimov parameter point should give chi2=0 for each scenario.
    """
    print_header("1. Fiducial self-consistency")

    passed = True

    for scenario in SCENARIOS:
        like = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **base_kwargs)
        )

        theta_fid = fiducial_theta(like)
        chi2 = like.chi2(theta_fid)

        ok = np.isclose(chi2, 0.0, rtol=0.0, atol=1e-10)
        passed &= ok

        print(f"{scenario:15s} chi2(fid) = {chi2:.12g}  {'OK' if ok else 'FAIL'}")

    return passed


def test_ratio_power_equivalence(base_kwargs):
    """
    Check that ratio and power observables produce the same chi2 surface after
    covariance transformations.
    """
    print_header("2. Ratio/power chi2 equivalence")

    passed = True

    for scenario in SCENARIOS:
        like_ratio = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="ratio", **base_kwargs)
        )
        like_power = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **base_kwargs)
        )

        print(f"\n{scenario}")

        for theta in default_theta_tests(like_power):
            cr = like_ratio.chi2(theta)
            cp = like_power.chi2(theta)
            diff = cr - cp

            ok = np.isclose(diff, 0.0, rtol=0.0, atol=1e-8)
            passed &= ok

            print(
                f"  theta={theta}  "
                f"chi2_ratio={cr:.12g}  chi2_power={cp:.12g}  "
                f"diff={diff:.3e}  {'OK' if ok else 'FAIL'}"
            )

    return passed


def test_scenario_hierarchy(base_kwargs):
    """
    For typical off-fiducial points, the expected information hierarchy is

        full_undamped > full_damped > linear_only

    in chi2. The fiducial point is excluded because all scenarios have chi2=0.
    """
    print_header("3. Scenario hierarchy")

    likes = {
        scenario: WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **base_kwargs)
        )
        for scenario in SCENARIOS
    }

    theta_tests = default_theta_tests(likes["full_damped"])[1:]

    passed = True

    for theta in theta_tests:
        chi = {scenario: likes[scenario].chi2(theta) for scenario in SCENARIOS}

        ok = (
            chi["full_undamped"] > chi["full_damped"]
            and chi["full_damped"] > chi["linear_only"]
        )
        passed &= ok

        print(
            f"\ntheta={theta}\n"
            f"  full_undamped = {chi['full_undamped']:.12g}\n"
            f"  full_damped   = {chi['full_damped']:.12g}\n"
            f"  linear_only   = {chi['linear_only']:.12g}\n"
            f"  {'OK' if ok else 'CHECK'}"
        )

    return passed


def test_covariance_transforms(base_kwargs):
    """
    Explicitly check:
        survey_sigma_ratio = survey_sigma_power / p_reference
        model_sigma_power  = epsilon * p_reference
        gp_sigma_power     = p_reference * gp_sigma_ratio
    """
    print_header("4. Covariance transformation checks")

    passed = True

    for scenario in SCENARIOS:
        like_ratio = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="ratio", **base_kwargs)
        )
        like_power = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **base_kwargs)
        )

        p_ref = concat_block_key(like_power.fid, "p_reference")

        survey_diff = max_abs(
            like_ratio.survey_sigma
            - like_power.survey_sigma / np.maximum(np.abs(p_ref), 1e-300)
        )

        model_ratio = like_ratio._model_error_sigma(like_ratio.fid)
        model_power = like_power._model_error_sigma(like_power.fid)

        model_diff = max_abs(
            model_ratio
            - model_power / np.maximum(np.abs(p_ref), 1e-300)
        )

        gp_diff = max_abs(
            like_ratio.fid["gp_std"]
            - like_power.fid["gp_std"] / np.maximum(np.abs(p_ref), 1e-300)
        )

        ok = (
            survey_diff < 1e-10
            and model_diff < 1e-12
            and gp_diff < 1e-12
        )
        passed &= ok

        print(
            f"\n{scenario}\n"
            f"  max |sigma_survey,R - sigma_survey,P/P_ref| = {survey_diff:.3e}\n"
            f"  max |sigma_model,R  - sigma_model,P/P_ref|  = {model_diff:.3e}\n"
            f"  max |sigma_GP,R     - sigma_GP,P/P_ref|     = {gp_diff:.3e}\n"
            f"  {'OK' if ok else 'FAIL'}"
        )

    return passed


def test_linear_only_cutoffs(base_kwargs):
    """
    Print the redshift-dependent kmax values used in the linear-only scenario.
    """
    print_header("5. Linear-only cutoff values")

    like = WiggleComparisonLikelihood(
        make_config(scenario="linear_only", observable="power", **base_kwargs)
    )

    print("k_fit_max_by_z:")
    for z, kmax in sorted(like.k_fit_max_by_z.items()):
        print(f"  z={z:g}: kmax={kmax:.8f} h/Mpc")

    print("\nBinned blocks:")
    for block in like.fid["blocks"]:
        dk = np.asarray(block.get("delta_k", np.diff(block["k"])), dtype=float)
        print(
            f"  z={block['z']:g}: "
            f"Nk={block['k'].size}, "
            f"k=[{block['k'].min():.6f}, {block['k'].max():.6f}], "
            f"dk_med={np.median(dk):.6f}"
        )

    return True


def test_undamped_damping_values(base_kwargs):
    """
    In the full_undamped scenario, damping should be identically one and Sigma
    should be zero.
    """
    print_header("6. Full-undamped internal damping check")

    like = WiggleComparisonLikelihood(
        make_config(scenario="full_undamped", observable="power", **base_kwargs)
    )

    passed = True

    for block in like.fid["blocks"]:
        damping = np.asarray(block["damping"], dtype=float)
        sigma = np.asarray(block["sigma"], dtype=float)

        damping_min = float(np.nanmin(damping))
        damping_max = float(np.nanmax(damping))
        sigma_max = float(np.nanmax(np.abs(sigma)))

        ok = (
            np.allclose(damping, 1.0, rtol=0.0, atol=1e-14)
            and np.allclose(sigma, 0.0, rtol=0.0, atol=1e-14)
        )
        passed &= ok

        print(
            f"z={block['z']:g}: "
            f"damping min/max={damping_min:.12g}/{damping_max:.12g}, "
            f"max |sigma|={sigma_max:.3e}  {'OK' if ok else 'FAIL'}"
        )

    return passed


def test_binning_vs_direct_sampling(base_kwargs):
    """
    Compare chi2 values for:
      - fine internal grid + top-hat binning to Delta k
      - direct coarse sampling with binning disabled

    This is not expected to be exactly identical. It checks whether the top-hat
    bandpower treatment materially changes the information.
    """
    print_header("7. Top-hat binning vs direct Delta-k sampling")

    passed = True

    # Top-hat binned, paper-facing setup.
    binned_kwargs = dict(base_kwargs)
    binned_kwargs["internal_npoints"] = max(int(base_kwargs["internal_npoints"]), 2000)
    binned_kwargs["bin_delta_k"] = float(base_kwargs["bin_delta_k"])

    # Direct coarse setup. Use roughly one point per Delta k over the fitted range.
    k_fit_min = float(base_kwargs["k_fit_min"])
    k_fit_max = float(base_kwargs["k_fit_max"])
    delta_k = float(base_kwargs["bin_delta_k"])
    n_direct = int(np.ceil((k_fit_max - k_fit_min) / delta_k)) + 20

    direct_kwargs = dict(base_kwargs)
    direct_kwargs["internal_npoints"] = n_direct
    direct_kwargs["bin_delta_k"] = 0.0

    print(f"direct internal_npoints = {n_direct}")

    for scenario in SCENARIOS:
        like_binned = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **binned_kwargs)
        )
        like_direct = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **direct_kwargs)
        )

        print(f"\n{scenario}")
        print(f"  binned vector length = {like_binned.data.size}")
        print(f"  direct vector length = {like_direct.data.size}")

        # Compare chi2 values at the same off-fiducial parameter points.
        for theta in default_theta_tests(like_binned)[1:]:
            cb = like_binned.chi2(theta)
            cd = like_direct.chi2(theta)

            ratio = cd / cb if cb != 0.0 else np.nan

            print(
                f"  theta={theta}: "
                f"chi2_binned={cb:.6g}, "
                f"chi2_direct={cd:.6g}, "
                f"direct/binned={ratio:.4g}"
            )

    # This is diagnostic rather than pass/fail.
    return passed


def test_internal_resolution_convergence(base_kwargs, npoints_values=(700, 1000, 2000, 4000)):
    """
    Compare chi2 values for different internal k-grid resolutions while keeping
    top-hat binning fixed.
    """
    print_header("8. Internal k-resolution convergence")

    passed = True

    reference_n = int(max(npoints_values))
    scenario = "full_damped"

    likes = {}
    for n in npoints_values:
        kwargs = dict(base_kwargs)
        kwargs["internal_npoints"] = int(n)
        kwargs["bin_delta_k"] = float(base_kwargs["bin_delta_k"])

        likes[int(n)] = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **kwargs)
        )

    theta_tests = default_theta_tests(likes[reference_n])[1:]

    for theta in theta_tests:
        cref = likes[reference_n].chi2(theta)

        print(f"\ntheta={theta}")
        print(f"  reference npoints={reference_n}, chi2={cref:.12g}")

        for n in npoints_values:
            c = likes[int(n)].chi2(theta)
            diff = c - cref
            frac = diff / cref if cref != 0.0 else np.nan

            ok = abs(frac) < 1e-2 or abs(diff) < 1e-6
            if n == reference_n:
                ok = True

            passed &= ok

            print(
                f"  npoints={int(n):5d}: "
                f"chi2={c:.12g}, diff={diff:.3e}, frac={frac:.3e} "
                f"{'OK' if ok else 'CHECK'}"
            )

    return passed


def test_paper_like_kmax_summary(base_kwargs, k_fit_max=0.25):
    """
    Build a paper-like kmax setup and print diagnostic quantities.

    This does not replace a full sampler run. It just checks the data-vector
    size, covariance scale and representative chi2 values for kmax=0.25.
    """
    print_header("9. Paper-like kmax diagnostic")

    kwargs = dict(base_kwargs)
    kwargs["k_fit_max"] = float(k_fit_max)
    kwargs["bin_delta_k"] = float(base_kwargs["bin_delta_k"])
    kwargs["internal_npoints"] = int(base_kwargs["internal_npoints"])

    for scenario in ("full_damped", "full_undamped"):
        like = WiggleComparisonLikelihood(
            make_config(scenario=scenario, observable="power", **kwargs)
        )

        print(f"\n{scenario}, k_fit_max={k_fit_max}")
        print(like.summary())

        for theta in default_theta_tests(like)[1:]:
            print(f"  theta={theta}, chi2={like.chi2(theta):.6g}")

    return True


def run_selected_tests(args):
    base_kwargs = dict(
        feature_type=args.feature_type,
        emulator_path=args.emulator,
        fid_A_feat=args.fid_A_feat,
        fid_omega=args.fid_omega,
        fid_phi=args.fid_phi,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        A_min=args.A_min,
        A_max=args.A_max,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        redshifts=tuple(args.redshifts),
        kmax=args.kmax,
        k_fit_min=args.k_fit_min,
        k_fit_max=args.k_fit_max,
        internal_npoints=args.internal_npoints,
        bin_delta_k=args.bin_delta_k,
        keep_partial_final_bin=args.keep_partial_final_bin,
        include_gp_uncertainty=args.include_gp_uncertainty,
        model_error_floor=args.model_error_floor,
        check_domain=not args.no_domain_check,
    )

    tests = {
        "fiducial": lambda: test_fiducial_self_consistency(base_kwargs),
        "ratio_power": lambda: test_ratio_power_equivalence(base_kwargs),
        "hierarchy": lambda: test_scenario_hierarchy(base_kwargs),
        "covariance": lambda: test_covariance_transforms(base_kwargs),
        "linear_cutoffs": lambda: test_linear_only_cutoffs(base_kwargs),
        "undamped": lambda: test_undamped_damping_values(base_kwargs),
        "binning": lambda: test_binning_vs_direct_sampling(base_kwargs),
        "resolution": lambda: test_internal_resolution_convergence(
            base_kwargs,
            npoints_values=tuple(args.resolution_npoints),
        ),
        "paper_kmax": lambda: test_paper_like_kmax_summary(
            base_kwargs,
            k_fit_max=args.paper_kmax,
        ),
    }

    selected = args.tests
    if selected == ["all"]:
        selected = list(tests.keys())

    results = {}

    for name in selected:
        if name not in tests:
            raise ValueError(
                f"Unknown test {name!r}. Available tests: {', '.join(tests)}"
            )

        try:
            results[name] = bool(tests[name]())
        except Exception as exc:
            results[name] = False
            print_header(f"{name} FAILED WITH EXCEPTION")
            print(repr(exc))
            if args.raise_on_fail:
                raise

    print_header("Summary")
    all_passed = True

    for name, passed in results.items():
        all_passed &= passed
        print(f"{name:15s}: {'PASS' if passed else 'CHECK/FAIL'}")

    if all_passed:
        print("\nAll selected checks passed.")
    else:
        print("\nSome selected checks failed or require inspection.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run diagnostic checks for the wiggle comparison likelihood."
    )

    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help=(
            "Tests to run. Use 'all' or any of: fiducial, ratio_power, "
            "hierarchy, covariance, linear_cutoffs, undamped, binning, "
            "resolution, paper_kmax."
        ),
    )

    parser.add_argument("--feature-type", choices=["log"], default="log")
    parser.add_argument("--emulator", type=Path, default=Path("emulators/log_sigma_gp.pkl"))

    parser.add_argument("--redshifts", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.65])

    parser.add_argument("--fid-A-feat", type=float, default=0.03)
    parser.add_argument("--fid-omega", type=float, default=1.26)
    parser.add_argument("--fid-phi", type=float, default=np.pi)

    parser.add_argument("--A-min", type=float, default=0.0)
    parser.add_argument("--A-max", type=float, default=0.06)

    parser.add_argument("--omega-min", type=float, default=0.8)
    parser.add_argument("--omega-max", type=float, default=2.0)

    parser.add_argument("--phi-min", type=float, default=0.0)
    parser.add_argument("--phi-max", type=float, default=2.0 * np.pi)

    parser.add_argument("--kmax", type=float, default=0.8)
    parser.add_argument("--k-fit-min", type=float, default=0.05)
    parser.add_argument("--k-fit-max", type=float, default=0.6)
    parser.add_argument("--internal-npoints", type=int, default=2000)
    parser.add_argument("--bin-delta-k", type=float, default=0.004)
    parser.add_argument("--keep-partial-final-bin", action="store_true")

    parser.add_argument("--include-gp-uncertainty", action="store_true")
    parser.add_argument("--model-error-floor", type=float, default=5e-3)

    parser.add_argument("--paper-kmax", type=float, default=0.25)
    parser.add_argument(
        "--resolution-npoints",
        type=int,
        nargs="+",
        default=[700, 1000, 2000, 4000],
    )

    parser.add_argument("--no-domain-check", action="store_true")
    parser.add_argument("--raise-on-fail", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_selected_tests(parse_args())