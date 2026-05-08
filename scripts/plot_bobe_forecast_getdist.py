#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np


DEFAULTS = {
    "log": {
        "input": Path("forecast_tests/BOBE_log_forecast_sampler_euclid_2d.npz"),
        "output": Path("forecast_tests/BOBE_log_forecast_getdist_euclid_2d.pdf"),
        "title": "Log feature forecast, Euclid GCsp, BOBE",
    },
    "linear": {
        "input": Path("forecast_tests/BOBE_linear_forecast_sampler_euclid_2d.npz"),
        "output": Path("forecast_tests/BOBE_linear_forecast_getdist_euclid_2d.pdf"),
        "title": "Linear feature forecast, Euclid GCsp, BOBE",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot BOBE forecast sampler output with GetDist."
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature template. Used to infer default input/output paths.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input NPZ file. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--phase-scale",
        choices=["rad", "twopi", "pi"],
        default="twopi",
        help="Plot phi as radians, phi/(2pi), or phi/pi.",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--weight-floor",
        type=float,
        default=0.0,
        help="Drop samples with weights <= this value.",
    )

    parser.add_argument(
        "--smooth-scale-1d",
        type=float,
        default=0.6,
        help="GetDist 1D smoothing scale.",
    )

    parser.add_argument(
        "--smooth-scale-2d",
        type=float,
        default=0.6,
        help="GetDist 2D smoothing scale.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    defaults = DEFAULTS[args.feature_type]

    input_path = args.input or defaults["input"]
    output_path = args.output or defaults["output"]
    title = args.title if args.title is not None else defaults["title"]

    return input_path, output_path, title


def phase_column(phi, phase_scale):
    phi = np.asarray(phi, dtype=float)

    if phase_scale == "rad":
        return phi, r"\phi"

    if phase_scale == "pi":
        return phi / np.pi, r"\phi/\pi"

    if phase_scale == "twopi":
        return phi / (2.0 * np.pi), r"\phi/(2\pi)"

    raise ValueError(f"Unknown phase_scale={phase_scale!r}.")


def as_scalar(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def load_bobe_npz(path, phase_scale, weight_floor):
    data = np.load(path, allow_pickle=True)

    samples_raw = np.asarray(data["samples"], dtype=float)
    weights = np.asarray(data["weights"], dtype=float)

    if samples_raw.ndim != 2:
        raise ValueError(f"Expected samples with shape (n_samples, ndim), got {samples_raw.shape}.")

    if samples_raw.shape[1] != 2:
        raise ValueError(
            "This plotting script expects 2D samples ordered as "
            "(omega_label, phi). "
            f"Got sample shape {samples_raw.shape}."
        )

    if weights.ndim != 1:
        weights = np.ravel(weights)

    if len(weights) != samples_raw.shape[0]:
        raise ValueError(
            f"Weight length {len(weights)} does not match number of samples "
            f"{samples_raw.shape[0]}."
        )

    omega = samples_raw[:, 0]
    phi, phi_label = phase_column(samples_raw[:, 1], phase_scale)

    finite = (
        np.isfinite(omega)
        & np.isfinite(phi)
        & np.isfinite(weights)
        & (weights > weight_floor)
    )

    omega = omega[finite]
    phi = phi[finite]
    weights = weights[finite]

    if omega.size == 0:
        raise ValueError("No finite positive-weight samples survived.")

    # GetDist accepts unnormalised weights, but normalising keeps summaries tidy.
    weight_sum = np.sum(weights)
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError("Weights do not have a finite positive sum.")

    weights = weights / weight_sum

    samples = np.column_stack([omega, phi])

    metadata = {
        "feature_type": str(as_scalar(data["feature_type"])) if "feature_type" in data else "unknown",
        "fid_A_feat": as_scalar(data["fid_A_feat"]) if "fid_A_feat" in data else np.nan,
        "fid_omega": as_scalar(data["fid_omega"]) if "fid_omega" in data else np.nan,
        "fid_phi": as_scalar(data["fid_phi"]) if "fid_phi" in data else np.nan,
        "logz": as_scalar(data["logz"]) if "logz" in data else np.nan,
        "logzerr": as_scalar(data["logzerr"]) if "logzerr" in data else np.nan,
        "phase_label": phi_label,
    }

    return samples, weights, metadata


def weighted_mean_and_std(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)

    mean = np.sum(w * x)
    var = np.sum(w * (x - mean) ** 2)

    return mean, np.sqrt(var)


def print_summary(samples, weights, metadata, input_path):
    omega = samples[:, 0]
    phi = samples[:, 1]

    omega_mean, omega_std = weighted_mean_and_std(omega, weights)
    phi_mean, phi_std = weighted_mean_and_std(phi, weights)

    ess = (np.sum(weights) ** 2) / np.sum(weights**2)

    print("\nBOBE sampler summary")
    print("--------------------")
    print(f"input:        {input_path}")
    print(f"n samples:    {samples.shape[0]}")
    print(f"ESS approx:   {ess:.1f}")
    print(f"feature_type: {metadata['feature_type']}")
    print(f"fid A_feat:   {metadata['fid_A_feat']}")
    print(f"fid omega:    {metadata['fid_omega']}")
    print(f"fid phi:      {metadata['fid_phi']}")

    try:
        print(f"logZ:         {float(metadata['logz']):.6g} +/- {float(metadata['logzerr']):.6g}")
    except Exception:
        print(f"logZ:         {metadata['logz']} +/- {metadata['logzerr']}")

    print("\nWeighted sample moments")
    print("-----------------------")
    print(f"omega_label:  {omega_mean:.6g} +/- {omega_std:.6g}")
    print(f"phi plotted:  {phi_mean:.6g} +/- {phi_std:.6g}")


def main():
    args = parse_args()
    input_path, output_path, title = resolve_defaults(args)

    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    samples, weights, metadata = load_bobe_npz(
        input_path,
        phase_scale=args.phase_scale,
        weight_floor=args.weight_floor,
    )

    print_summary(samples, weights, metadata, input_path)

    names = ["omega_label", "phi"]
    labels = [r"\omega_{\rm label}", metadata["phase_label"]]

    ranges = {
        "omega_label": (float(np.nanmin(samples[:, 0])), float(np.nanmax(samples[:, 0]))),
        "phi": (float(np.nanmin(samples[:, 1])), float(np.nanmax(samples[:, 1]))),
    }

    gd = MCSamples(
        samples=samples,
        weights=weights,
        names=names,
        labels=labels,
        ranges=ranges,
        name_tag="Forecast",
    )

    gd.updateSettings(
        {
            "contours": [0.68, 0.95],
            "smooth_scale_1D": args.smooth_scale_1d,
            "smooth_scale_2D": args.smooth_scale_2d,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = plots.get_subplot_plotter(width_inch=7.0)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.settings.legend_fontsize = 12
    g.settings.figure_legend_frame = False

    g.triangle_plot(
        [gd],
        names,
        filled=True,
        title_limit=1,
    )

    if title:
        plt.suptitle(title, fontsize=14, y=0.98)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()