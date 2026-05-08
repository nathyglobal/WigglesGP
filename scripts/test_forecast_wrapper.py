#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np

from wigglesgp.emulator import SigmaEmulator
from wigglesgp.damping import damped_wiggle_ratio
from wigglesgp.power import nonlinear_wiggle_power
from wigglesgp.forecasting import damped_feature_ratio_from_emulator


DEFAULTS = {
    "log": {
        "emulator": Path("emulators/log_sigma_gp.pkl"),
        "omega": 1.26,
    },
    "linear": {
        "emulator": Path("emulators/linear_sigma_gp.pkl"),
        "omega": 0.87,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the forecast-facing damping wrapper using a saved Sigma "
            "emulator and toy power spectra."
        )
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature type to test.",
    )

    parser.add_argument(
        "--emulator",
        type=Path,
        default=None,
        help="Path to saved Sigma emulator. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--z",
        type=float,
        default=1.0,
        help="Redshift at which to evaluate the damping correction.",
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help=(
            "Stored emulator frequency coordinate. For the current calibration "
            "this is the omega label, i.e. log10(omega), not 10**omega. "
            "Defaults from --feature-type."
        ),
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.03,
        help="Feature amplitude.",
    )

    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Feature phase in radians.",
    )

    parser.add_argument(
        "--k-min",
        type=float,
        default=0.05,
        help="Minimum k value.",
    )

    parser.add_argument(
        "--k-max",
        type=float,
        default=0.6,
        help="Maximum k value.",
    )

    parser.add_argument(
        "--n-k",
        type=int,
        default=200,
        help="Number of k points.",
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0.67,
        help="Reduced Hubble parameter used by the feature convention.",
    )

    parser.add_argument(
        "--k-pivot",
        type=float,
        default=0.05,
        help="Feature pivot scale.",
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
    omega = args.omega if args.omega is not None else defaults["omega"]

    return emulator_path, omega


def toy_vanilla_linear_power(k):
    """
    Smooth toy linear matter power spectrum.

    This is not a physical forecast model. It only exists to verify that the
    damping/power-spectrum wiring behaves correctly.
    """
    k = np.asarray(k, dtype=float)
    return k**0.96 * np.exp(-2.0 * k)


def toy_nonlinear_boost(k):
    """
    Smooth toy non-linear boost applied to the vanilla linear spectrum.

    Again, this is only a wiring test.
    """
    k = np.asarray(k, dtype=float)
    return 1.0 + 0.4 * (k / 0.6) ** 2


def main():
    args = parse_args()

    emulator_path, omega_label = resolve_defaults(args)

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator file does not exist: {emulator_path}")

    check_domain = not args.no_domain_check

    emulator = SigmaEmulator.from_file(emulator_path)

    k = np.linspace(args.k_min, args.k_max, args.n_k)

    z_grid = np.full_like(k, args.z, dtype=float)
    omega_label_grid = np.full_like(k, omega_label, dtype=float)

    omega_model = 10.0 ** omega_label

    # ------------------------------------------------------------------
    # Build toy spectra.
    # ------------------------------------------------------------------
    p_van_lin = toy_vanilla_linear_power(k)
    p_van_nl = p_van_lin * toy_nonlinear_boost(k)

    # Undamped linear feature ratio using the existing public model.
    # sigma=0 means no non-linear damping.
    ratio_wiggle_lin = damped_wiggle_ratio(
        k,
        omega_model,
        args.phase,
        0.0,
        feature_type=args.feature_type,
        amplitude=args.amplitude,
        k_pivot=args.k_pivot,
        h=args.h,
    )

    p_wig_lin = p_van_lin * ratio_wiggle_lin

    # Emulator-predicted damping envelope.
    damping, sigma = emulator.damping(
        k,
        z_grid,
        omega_label_grid,
        return_sigma=True,
        check_domain=check_domain,
    )

    # Forecast-facing non-linear wiggle power.
    p_wig_nl = nonlinear_wiggle_power(
        p_van_lin=p_van_lin,
        p_van_nl=p_van_nl,
        p_wig_lin=p_wig_lin,
        damping=damping,
    )

    ratio_wiggle_nl = p_wig_nl / p_van_nl

    # Same ratio via the public forecasting wrapper.
    ratio_wrapper, components = damped_feature_ratio_from_emulator(
        k,
        z_grid,
        omega_label_grid,
        emulator,
        feature_type=args.feature_type,
        amplitude=args.amplitude,
        phase=args.phase,
        k_pivot=args.k_pivot,
        h=args.h,
        check_domain=check_domain,
        return_components=True,
    )

    # Direct equivalent using damped_wiggle_ratio with sigma from the emulator.
    # This should match p_wig_nl / p_van_nl for this toy construction.
    ratio_direct = damped_wiggle_ratio(
        k,
        omega_model,
        args.phase,
        sigma,
        feature_type=args.feature_type,
        amplitude=args.amplitude,
        k_pivot=args.k_pivot,
        h=args.h,
    )

    direct_diff = ratio_wiggle_nl - ratio_direct
    wrapper_diff = ratio_wiggle_nl - ratio_wrapper
    linear_diff = ratio_wiggle_lin - components["ratio_linear"]
    damping_diff = damping - components["damping"]
    sigma_diff = sigma - components["sigma"]

    print("\nForecast-wrapper test")
    print("---------------------")
    print(f"emulator:        {emulator_path}")
    print(f"feature_type:    {args.feature_type}")
    print(f"z:               {args.z}")
    print(f"omega_label:     {omega_label}")
    print(f"omega_model:     {omega_model:.6g}")
    print(f"amplitude:       {args.amplitude}")
    print(f"phase:           {args.phase}")
    print(f"k range:         {k[0]:.6g} -> {k[-1]:.6g}")
    print(f"n_k:             {len(k)}")

    print("\nSigma and damping")
    print("-----------------")
    print(f"sigma min/max:   {np.nanmin(sigma):.6g} / {np.nanmax(sigma):.6g}")
    print(f"damping min/max: {np.nanmin(damping):.6g} / {np.nanmax(damping):.6g}")

    print("\nRatios")
    print("------")
    print(
        "linear feature ratio min/max: "
        f"{np.nanmin(ratio_wiggle_lin):.6g} / {np.nanmax(ratio_wiggle_lin):.6g}"
    )
    print(
        "damped feature ratio min/max: "
        f"{np.nanmin(ratio_wiggle_nl):.6g} / {np.nanmax(ratio_wiggle_nl):.6g}"
    )

    print("\nInternal consistency")
    print("--------------------")
    print(
        "max |power-wrapper ratio - direct damped ratio|: "
        f"{np.nanmax(np.abs(direct_diff)):.6e}"
    )
    print(
        "rms |power-wrapper ratio - direct damped ratio|: "
        f"{np.sqrt(np.nanmean(direct_diff**2)):.6e}"
    )
    print(
        "max |power-wrapper ratio - forecasting wrapper ratio|: "
        f"{np.nanmax(np.abs(wrapper_diff)):.6e}"
    )
    print(
        "rms |power-wrapper ratio - forecasting wrapper ratio|: "
        f"{np.sqrt(np.nanmean(wrapper_diff**2)):.6e}"
    )
    print(
        "max |manual linear ratio - wrapper linear ratio|: "
        f"{np.nanmax(np.abs(linear_diff)):.6e}"
    )
    print(
        "max |manual damping - wrapper damping|: "
        f"{np.nanmax(np.abs(damping_diff)):.6e}"
    )
    print(
        "max |manual sigma - wrapper sigma|: "
        f"{np.nanmax(np.abs(sigma_diff)):.6e}"
    )

    print("\nSample values")
    print("-------------")

    sample_indices = np.linspace(0, len(k) - 1, 8, dtype=int)

    for idx in sample_indices:
        print(
            f"k={k[idx]:.4f}, "
            f"sigma={sigma[idx]:.6g}, "
            f"D={damping[idx]:.6g}, "
            f"R_lin={ratio_wiggle_lin[idx]:.6g}, "
            f"R_nl={ratio_wiggle_nl[idx]:.6g}, "
            f"R_direct={ratio_direct[idx]:.6g}, "
            f"R_wrapper={ratio_wrapper[idx]:.6g}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()