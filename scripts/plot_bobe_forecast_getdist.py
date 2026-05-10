#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np


DEFAULTS = {
    "log": {
        "input": Path("forecast_tests/BOBE_log_forecast_sampler_euclid_2d.npz"),
        "output": Path("FinalPlots/forecast_getdist_log_euclid_2d.pdf"),
        "title": r"Euclid GCsp forecast: logarithmic features",
    },
    "linear": {
        "input": Path("forecast_tests/BOBE_linear_forecast_sampler_euclid_2d.npz"),
        "output": Path("FinalPlots/forecast_getdist_linear_euclid_2d.pdf"),
        "title": r"Euclid GCsp forecast: linear features",
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

    parser.add_argument(
        "--width-inch",
        type=float,
        default=6.6,
        help="GetDist plot width in inches.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )

    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering.",
    )

    parser.add_argument(
        "--no-fiducial-marker",
        action="store_true",
        help="Do not mark the injected fiducial point.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    defaults = DEFAULTS[args.feature_type]

    input_path = args.input or defaults["input"]
    output_path = args.output or defaults["output"]
    title = args.title if args.title is not None else defaults["title"]

    return input_path, output_path, title


def set_plot_style(use_latex=True):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.labelsize": 15,
            "axes.titlesize": 20,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def phase_column(phi, phase_scale):
    phi = np.asarray(phi, dtype=float)

    if phase_scale == "rad":
        return phi, r"\phi"

    if phase_scale == "pi":
        return phi / np.pi, r"\phi/\pi"

    if phase_scale == "twopi":
        return phi / (2.0 * np.pi), r"\phi/2\pi"

    raise ValueError(f"Unknown phase_scale={phase_scale!r}.")


def phase_scalar(phi, phase_scale):
    phi = float(phi)

    if phase_scale == "rad":
        return phi

    if phase_scale == "pi":
        return phi / np.pi

    if phase_scale == "twopi":
        return phi / (2.0 * np.pi)

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
        raise ValueError(
            f"Expected samples with shape (n_samples, ndim), got {samples_raw.shape}."
        )

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

    weight_sum = np.sum(weights)
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError("Weights do not have a finite positive sum.")

    weights = weights / weight_sum

    samples = np.column_stack([omega, phi])

    fid_phi_raw = as_scalar(data["fid_phi"]) if "fid_phi" in data else np.nan
    fid_phi_plot = (
        phase_scalar(fid_phi_raw, phase_scale)
        if np.isfinite(fid_phi_raw)
        else np.nan
    )

    metadata = {
        "feature_type": str(as_scalar(data["feature_type"])) if "feature_type" in data else "unknown",
        "fid_A_feat": as_scalar(data["fid_A_feat"]) if "fid_A_feat" in data else np.nan,
        "fid_omega": as_scalar(data["fid_omega"]) if "fid_omega" in data else np.nan,
        "fid_phi": fid_phi_raw,
        "fid_phi_plot": fid_phi_plot,
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
    print(f"fid phi raw:  {metadata['fid_phi']}")
    print(f"fid phi plot: {metadata['fid_phi_plot']}")

    try:
        print(
            f"logZ:         {float(metadata['logz']):.6g} "
            f"+/- {float(metadata['logzerr']):.6g}"
        )
    except Exception:
        print(f"logZ:         {metadata['logz']} +/- {metadata['logzerr']}")

    print("\nWeighted sample moments")
    print("-----------------------")
    print(f"omega_label:  {omega_mean:.6g} +/- {omega_std:.6g}")
    print(f"phi plotted:  {phi_mean:.6g} +/- {phi_std:.6g}")


def padded_range(values, fid_value=None, pad_fraction=0.06):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if fid_value is not None and np.isfinite(fid_value):
        finite = np.concatenate([finite, [float(fid_value)]])

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    if np.isclose(vmin, vmax):
        pad = 0.05 * max(abs(vmin), 1.0)
    else:
        pad = pad_fraction * (vmax - vmin)

    return vmin - pad, vmax + pad


def add_fiducial_marker(g, metadata):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fid_omega = float(metadata["fid_omega"])
    fid_phi = float(metadata["fid_phi_plot"])

    if not (np.isfinite(fid_omega) and np.isfinite(fid_phi)):
        return

    axes = np.asarray(g.subplots)

    # Triangle plot for two parameters has axes:
    # [0,0] omega 1D
    # [1,0] phi-vs-omega 2D
    # [1,1] phi 1D
    if axes.shape[0] >= 2 and axes.shape[1] >= 2:
        ax_omega_1d = axes[0, 0]
        ax_2d = axes[1, 0]
        ax_phi_1d = axes[1, 1]

        ax_omega_1d.axvline(
            fid_omega,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )

        ax_phi_1d.axvline(
            fid_phi,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )

        ax_2d.axvline(
            fid_omega,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        ax_2d.axhline(
            fid_phi,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        ax_2d.plot(
            fid_omega,
            fid_phi,
            marker="x",
            color="black",
            markersize=8,
            markeredgewidth=1.5,
            linestyle="none",
            zorder=10,
        )

        handle = Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            marker="x",
            markersize=7,
            markeredgewidth=1.3,
            linewidth=1.0,
            label="Injected fiducial",
        )

        ax_2d.legend(
            handles=[handle],
            loc="upper right",
            frameon=False,
            fontsize=11,
        )


def main():
    args = parse_args()
    input_path, output_path, title = resolve_defaults(args)

    set_plot_style(use_latex=not args.no_latex)

    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    samples, weights, metadata = load_bobe_npz(
        input_path,
        phase_scale=args.phase_scale,
        weight_floor=args.weight_floor,
    )

    print_summary(samples, weights, metadata, input_path)

    names = ["omega_label", "phi"]
    labels = [r"\log_{10}(\omega)", metadata["phase_label"]]

    ranges = {
        "omega_label": padded_range(samples[:, 0], metadata["fid_omega"]),
        "phi": padded_range(samples[:, 1], metadata["fid_phi_plot"]),
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

    g = plots.get_subplot_plotter(width_inch=args.width_inch)

    g.settings.axes_fontsize = 20
    g.settings.lab_fontsize = 20
    g.settings.legend_fontsize = 11
    g.settings.figure_legend_frame = False
    g.settings.linewidth = 1.2
    g.settings.linewidth_contour = 1.0
    g.settings.axis_marker_lw = 1.0
    g.settings.title_limit_fontsize = 20

    g.triangle_plot(
        [gd],
        names,
        filled=True,
        title_limit=1,
        legend_labels=None,
    )

    if not args.no_fiducial_marker:
        add_fiducial_marker(g, metadata)

    #if title:
    #    plt.suptitle(title, fontsize=15, y=0.985)

    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()