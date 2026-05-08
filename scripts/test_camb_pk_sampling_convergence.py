#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.power import nonlinear_wiggle_power
from wigglesgp.camb_power import get_vanilla_and_wiggle_spectra


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test convergence of CAMB primordial spline settings for "
            "oscillatory feature matter spectra."
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
        "--z",
        type=float,
        default=1.0,
        help="Redshift.",
    )

    parser.add_argument(
        "--log10omega",
        type=float,
        default=2.0,
        help="Feature frequency label in the emulator convention.",
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
        "--pk-N-min-values",
        type=int,
        nargs="+",
        default=[4000, 8000, 16000, 24000],
        help="CAMB primordial spline N_min values to test.",
    )

    parser.add_argument(
        "--pk-rtol-values",
        type=float,
        nargs="+",
        default=[1e-10, 1e-12],
        help="CAMB primordial spline rtol values to test.",
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
        default=900,
        help="Number of CAMB matter-power k points.",
    )

    parser.add_argument(
        "--k-fit-min",
        type=float,
        default=0.05,
        help="Minimum k/h used for diagnostics.",
    )

    parser.add_argument(
        "--k-fit-max",
        type=float,
        default=0.6,
        help="Maximum k/h used for diagnostics.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forecast_tests/pk_sampling_convergence"),
        help="Output directory.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


def set_latex_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.unicode_minus": False,
        }
    )


def sampling_label(pk_N_min, pk_rtol):
    return f"N{int(pk_N_min)}_rtol{pk_rtol:.0e}".replace("-", "m")


def compute_case(
    *,
    emulator,
    feature_type,
    z,
    log10omega,
    A_feat,
    phi,
    pk_N_min,
    pk_rtol,
    kmax,
    npoints,
    k_fit_min,
    k_fit_max,
    check_domain,
):
    spectra = get_vanilla_and_wiggle_spectra(
        redshift=z,
        log10omega_feat=log10omega,
        A_feat=A_feat,
        phi=phi,
        feature_type=feature_type,
        kmax=kmax,
        npoints=npoints,
        camb_options={
            "pk_N_min": int(pk_N_min),
            "pk_rtol": float(pk_rtol),
        },
    )

    k = spectra["k"]
    mask = (k >= k_fit_min) & (k <= k_fit_max)

    if not np.any(mask):
        raise ValueError(
            f"No CAMB k values inside requested range [{k_fit_min}, {k_fit_max}]."
        )

    k = k[mask]
    p_van_lin = spectra["p_van_lin"][mask]
    p_van_nl = spectra["p_van_nl"][mask]
    p_wig_lin = spectra["p_wig_lin"][mask]

    z_grid = np.full_like(k, float(z))
    omega_grid = np.full_like(k, float(log10omega))

    damping, sigma = emulator.damping(
        k,
        z_grid,
        omega_grid,
        return_sigma=True,
        check_domain=check_domain,
    )

    p_wig_nl = nonlinear_wiggle_power(
        p_van_lin=p_van_lin,
        p_van_nl=p_van_nl,
        p_wig_lin=p_wig_lin,
        damping=damping,
    )

    ratio_lin = p_wig_lin / p_van_lin
    ratio_nl = p_wig_nl / p_van_nl

    return {
        "k": k,
        "ratio_lin": ratio_lin,
        "ratio_nl": ratio_nl,
        "damping": damping,
        "sigma": sigma,
        "p_van_lin": p_van_lin,
        "p_van_nl": p_van_nl,
        "p_wig_lin": p_wig_lin,
        "p_wig_nl": p_wig_nl,
        "pk_N_min": int(pk_N_min),
        "pk_rtol": float(pk_rtol),
    }


def interpolate_to_reference(k_ref, k, y):
    if len(k) == len(k_ref) and np.allclose(k, k_ref, rtol=0.0, atol=0.0):
        return y

    return np.interp(k_ref, k, y)


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "pk_N_min",
        "pk_rtol",
        "lin_max_abs_diff",
        "lin_rms_diff",
        "nl_max_abs_diff",
        "nl_rms_diff",
        "lin_max_frac_diff",
        "nl_max_frac_diff",
    ]

    with open(path, "w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(f"{row[col]:.12g}" for col in columns) + "\n")


def plot_cases(path, cases, reference_key):
    set_latex_style()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.5),
        sharex=True,
    )

    ax_lin, ax_nl = axes

    for key, case in cases.items():
        k = case["k"]
        label = key

        linestyle = "-" if key == reference_key else "--"

        ax_lin.plot(
            k,
            case["ratio_lin"],
            linewidth=1.2,
            linestyle=linestyle,
            label=label,
        )

        ax_nl.plot(
            k,
            case["ratio_nl"],
            linewidth=1.2,
            linestyle=linestyle,
            label=label,
        )

    for ax in axes:
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=13,
        )
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    ax_lin.set_ylabel(r"$R_{\rm lin}(k)$", fontsize=16)
    ax_nl.set_ylabel(r"$R_{\rm emu}(k)$", fontsize=16)
    ax_nl.set_xlabel(r"$k\,[h/{\rm Mpc}]$", fontsize=16)

    ax_lin.set_title("CAMB primordial spline convergence", fontsize=18)

    handles, labels = ax_lin.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.subplots_adjust(hspace=0.0, top=0.90)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_differences(path, cases, reference_key):
    set_latex_style()

    ref = cases[reference_key]
    k_ref = ref["k"]
    lin_ref = ref["ratio_lin"]
    nl_ref = ref["ratio_nl"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.5),
        sharex=True,
    )

    ax_lin, ax_nl = axes

    for key, case in cases.items():
        if key == reference_key:
            continue

        lin = interpolate_to_reference(k_ref, case["k"], case["ratio_lin"])
        nl = interpolate_to_reference(k_ref, case["k"], case["ratio_nl"])

        ax_lin.plot(
            k_ref,
            lin - lin_ref,
            linewidth=1.2,
            label=key,
        )

        ax_nl.plot(
            k_ref,
            nl - nl_ref,
            linewidth=1.2,
            label=key,
        )

    for ax in axes:
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=13,
        )
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    ax_lin.set_ylabel(r"$R_{\rm lin}-R_{\rm ref}$", fontsize=16)
    ax_nl.set_ylabel(r"$R_{\rm emu}-R_{\rm ref}$", fontsize=16)
    ax_nl.set_xlabel(r"$k\,[h/{\rm Mpc}]$", fontsize=16)

    ax_lin.set_title("Difference relative to highest-resolution setting", fontsize=18)

    handles, labels = ax_lin.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, 1.01),
        )

    fig.subplots_adjust(hspace=0.0, top=0.90)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    check_domain = not args.no_domain_check

    args.output_dir.mkdir(parents=True, exist_ok=True)

    emulator = SigmaEmulator.from_file(args.emulator)

    cases = {}

    print("\nCAMB primordial spline convergence test")
    print("---------------------------------------")
    print(f"feature_type: {args.feature_type}")
    print(f"z:            {args.z}")
    print(f"log10omega:   {args.log10omega}")
    print(f"A_feat:       {args.A_feat}")
    print(f"phi:          {args.phi}")
    print(f"output_dir:   {args.output_dir}")

    total = len(args.pk_N_min_values) * len(args.pk_rtol_values)
    counter = 0

    for pk_N_min in args.pk_N_min_values:
        for pk_rtol in args.pk_rtol_values:
            counter += 1
            key = sampling_label(pk_N_min, pk_rtol)

            print(f"\n[{counter}/{total}] {key}")

            case = compute_case(
                emulator=emulator,
                feature_type=args.feature_type,
                z=args.z,
                log10omega=args.log10omega,
                A_feat=args.A_feat,
                phi=args.phi,
                pk_N_min=pk_N_min,
                pk_rtol=pk_rtol,
                kmax=args.kmax,
                npoints=args.npoints,
                k_fit_min=args.k_fit_min,
                k_fit_max=args.k_fit_max,
                check_domain=check_domain,
            )

            cases[key] = case

            print(
                f"  Rlin={np.nanmin(case['ratio_lin']):.6g}/"
                f"{np.nanmax(case['ratio_lin']):.6g}, "
                f"Rnl={np.nanmin(case['ratio_nl']):.6g}/"
                f"{np.nanmax(case['ratio_nl']):.6g}"
            )

    # Use the highest N_min and smallest rtol as the reference.
    reference_N = max(args.pk_N_min_values)
    reference_rtol = min(args.pk_rtol_values)
    reference_key = sampling_label(reference_N, reference_rtol)

    if reference_key not in cases:
        raise RuntimeError(f"Reference case {reference_key} was not generated.")

    ref = cases[reference_key]
    k_ref = ref["k"]
    lin_ref = ref["ratio_lin"]
    nl_ref = ref["ratio_nl"]

    summary_rows = []

    print("\nConvergence relative to reference")
    print("---------------------------------")
    print(f"reference: {reference_key}")

    for key, case in cases.items():
        lin = interpolate_to_reference(k_ref, case["k"], case["ratio_lin"])
        nl = interpolate_to_reference(k_ref, case["k"], case["ratio_nl"])

        lin_diff = lin - lin_ref
        nl_diff = nl - nl_ref

        lin_frac = lin_diff / lin_ref
        nl_frac = nl_diff / nl_ref

        row = {
            "pk_N_min": float(case["pk_N_min"]),
            "pk_rtol": float(case["pk_rtol"]),
            "lin_max_abs_diff": float(np.nanmax(np.abs(lin_diff))),
            "lin_rms_diff": float(np.sqrt(np.nanmean(lin_diff**2))),
            "nl_max_abs_diff": float(np.nanmax(np.abs(nl_diff))),
            "nl_rms_diff": float(np.sqrt(np.nanmean(nl_diff**2))),
            "lin_max_frac_diff": float(np.nanmax(np.abs(lin_frac))),
            "nl_max_frac_diff": float(np.nanmax(np.abs(nl_frac))),
        }

        summary_rows.append(row)

        print(
            f"{key}: "
            f"lin max={row['lin_max_abs_diff']:.3e}, "
            f"lin rms={row['lin_rms_diff']:.3e}, "
            f"nl max={row['nl_max_abs_diff']:.3e}, "
            f"nl rms={row['nl_rms_diff']:.3e}"
        )

    summary_path = args.output_dir / "pk_sampling_convergence_summary.csv"
    plot_path = args.output_dir / "pk_sampling_convergence_ratios.pdf"
    diff_path = args.output_dir / "pk_sampling_convergence_differences.pdf"

    write_summary(summary_path, summary_rows)
    plot_cases(plot_path, cases, reference_key)
    plot_differences(diff_path, cases, reference_key)

    print("\nDone.")
    print(f"Wrote summary:     {summary_path}")
    print(f"Wrote ratio plot:  {plot_path}")
    print(f"Wrote diff plot:   {diff_path}")


if __name__ == "__main__":
    main()