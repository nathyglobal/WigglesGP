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
            "Generate CAMB vanilla/feature spectra and apply the saved "
            "Sigma-emulator non-linear damping correction."
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
        default=1.26,
        help=(
            "Feature frequency label in the emulator convention. "
            "This is log10(omega), not omega."
        ),
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
        "--output",
        type=Path,
        default=Path("forecast_tests/log_camb_emulator_power_z1_w126.csv"),
        help="CSV output path.",
    )

    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="Disable emulator domain checks.",
    )

    return parser.parse_args()


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


def main():
    args = parse_args()
    check_domain = not args.no_domain_check

    emulator = SigmaEmulator.from_file(args.emulator)

    spectra = get_vanilla_and_wiggle_spectra(
        redshift=args.z,
        log10omega_feat=args.log10omega,
        A_feat=args.A_feat,
        phi=args.phi,
        feature_type=args.feature_type,
        kmax=args.kmax,
        npoints=args.npoints,
    )

    k = spectra["k"]
    p_van_lin = spectra["p_van_lin"]
    p_van_nl = spectra["p_van_nl"]
    p_wig_lin = spectra["p_wig_lin"]

    mask = (k >= args.k_fit_min) & (k <= args.k_fit_max)

    k_fit = k[mask]
    p_van_lin_fit = p_van_lin[mask]
    p_van_nl_fit = p_van_nl[mask]
    p_wig_lin_fit = p_wig_lin[mask]

    z_grid = np.full_like(k_fit, args.z, dtype=float)
    omega_grid = np.full_like(k_fit, args.log10omega, dtype=float)

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

    write_output_csv(args.output, output_data)

    print("\nCAMB + Sigma-emulator forecast test")
    print("-----------------------------------")
    print(f"emulator:       {args.emulator}")
    print(f"feature_type:   {args.feature_type}")
    print(f"z:              {args.z}")
    print(f"log10omega:     {args.log10omega}")
    print(f"omega:          {10.0 ** args.log10omega:.6g}")
    print(f"A_feat:         {args.A_feat}")
    print(f"phi:            {args.phi}")
    print(f"CAMB kmax:      {args.kmax}")
    print(f"CAMB npoints:   {args.npoints}")
    print(f"fit k range:    {args.k_fit_min} -> {args.k_fit_max}")
    print(f"n fit points:   {len(k_fit)}")
    print(f"output:         {args.output}")

    print("\nSigma and damping")
    print("-----------------")
    print(f"sigma min/max:   {np.nanmin(sigma):.6g} / {np.nanmax(sigma):.6g}")
    print(f"damping min/max: {np.nanmin(damping):.6g} / {np.nanmax(damping):.6g}")

    print("\nPower-spectrum ratios")
    print("---------------------")
    print(
        "linear wiggle ratio min/max: "
        f"{np.nanmin(ratio_lin):.6g} / {np.nanmax(ratio_lin):.6g}"
    )
    print(
        "emulator-damped ratio min/max: "
        f"{np.nanmin(ratio_nl):.6g} / {np.nanmax(ratio_nl):.6g}"
    )
    print(
        "non-linear vanilla boost min/max: "
        f"{np.nanmin(p_van_nl_fit / p_van_lin_fit):.6g} / "
        f"{np.nanmax(p_van_nl_fit / p_van_lin_fit):.6g}"
    )

    print("\nSample values")
    print("-------------")

    sample_indices = np.linspace(0, len(k_fit) - 1, 8, dtype=int)

    for idx in sample_indices:
        print(
            f"k={k_fit[idx]:.4f}, "
            f"sigma={sigma[idx]:.6g}, "
            f"D={damping[idx]:.6g}, "
            f"R_lin={ratio_lin[idx]:.6g}, "
            f"R_nl={ratio_nl[idx]:.6g}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()