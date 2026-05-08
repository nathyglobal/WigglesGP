#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a forecast likelihood grid into weighted GetDist samples "
            "and make a corner plot."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("forecast_tests/log_forecast_grid_full_domain_Afixed.csv"),
        help="Input forecast grid CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forecast_tests/log_forecast_getdist_corner.pdf"),
        help="Output corner plot path.",
    )

    parser.add_argument(
        "--include-A",
        action="store_true",
        help="Include A_feat in the GetDist samples if more than one A value is present.",
    )

    parser.add_argument(
        "--phase-scale",
        choices=["rad", "twopi", "pi"],
        default="twopi",
        help=(
            "How to plot phase. 'rad' plots phi, 'twopi' plots phi/(2pi), "
            "and 'pi' plots phi/pi."
        ),
    )

    parser.add_argument(
        "--weight-floor",
        type=float,
        default=1e-300,
        help="Minimum retained unnormalised weight.",
    )

    parser.add_argument(
        "--thin-zero-weight",
        action="store_true",
        help="Drop samples whose likelihood weight underflows below the floor.",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )

    return parser.parse_args()


def load_grid(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    required = ["A_feat", "log10omega", "phi", "delta_chi2"]
    for name in required:
        if name not in data.dtype.names:
            raise ValueError(
                f"Missing column {name!r} in {path}. "
                f"Available columns are {data.dtype.names}."
            )

    return data


def phase_column(phi, phase_scale):
    phi = np.asarray(phi, dtype=float)

    if phase_scale == "rad":
        return phi, r"\phi"

    if phase_scale == "pi":
        return phi / np.pi, r"\phi/\pi"

    if phase_scale == "twopi":
        return phi / (2.0 * np.pi), r"\phi/(2\pi)"

    raise ValueError(f"Unknown phase_scale={phase_scale!r}.")


def make_getdist_samples(data, *, include_A=False, phase_scale="twopi", weight_floor=1e-300, thin_zero_weight=False):
    from getdist import MCSamples

    A = np.asarray(data["A_feat"], dtype=float)
    log10omega = np.asarray(data["log10omega"], dtype=float)
    phi_plot, phi_label = phase_column(data["phi"], phase_scale)

    delta_chi2 = np.asarray(data["delta_chi2"], dtype=float)
    delta_chi2 = delta_chi2 - np.nanmin(delta_chi2)

    weights = np.exp(-0.5 * delta_chi2)

    if thin_zero_weight:
        mask = np.isfinite(weights) & (weights > weight_floor)
    else:
        weights = np.maximum(weights, weight_floor)
        mask = np.isfinite(weights)

    if include_A and len(np.unique(A)) > 1:
        samples = np.column_stack([A[mask], log10omega[mask], phi_plot[mask]])
        names = ["A_feat", "log10omega", "phi"]
        labels = [r"A_{\rm feat}", r"\log_{10}\omega", phi_label]
    else:
        samples = np.column_stack([log10omega[mask], phi_plot[mask]])
        names = ["log10omega", "phi"]
        labels = [r"\log_{10}\omega", phi_label]

    weights = weights[mask]

    if samples.shape[0] == 0:
        raise ValueError("No finite-weight samples survived.")

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        names=names,
        labels=labels,
        name_tag="Forecast",
    )

    gd_samples.updateSettings(
        {
            "contours": [0.68, 0.95],
            "smooth_scale_2D": 0.6,
            "smooth_scale_1D": 0.6,
        }
    )

    return gd_samples


def print_summary(data):
    delta = np.asarray(data["delta_chi2"], dtype=float)
    best = np.nanargmin(delta)

    print("\nGrid summary")
    print("------------")
    print(f"n points:          {len(delta)}")
    print(f"min delta_chi2:    {np.nanmin(delta):.6g}")
    print(f"max delta_chi2:    {np.nanmax(delta):.6g}")
    print(f"best A_feat:       {data['A_feat'][best]:.6g}")
    print(f"best log10omega:   {data['log10omega'][best]:.6g}")
    print(f"best phi:          {data['phi'][best]:.6g}")


def main():
    args = parse_args()

    import matplotlib.pyplot as plt
    from getdist import plots

    data = load_grid(args.input)
    print_summary(data)

    samples = make_getdist_samples(
        data,
        include_A=args.include_A,
        phase_scale=args.phase_scale,
        weight_floor=args.weight_floor,
        thin_zero_weight=args.thin_zero_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    g = plots.get_subplot_plotter(width_inch=7.0)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.settings.legend_fontsize = 12
    g.settings.figure_legend_frame = False

    params = samples.getParamNames().list()

    g.triangle_plot(
        [samples],
        params,
        filled=True,
    )

    if args.title is not None:
        plt.suptitle(args.title, fontsize=14, y=0.98)

    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()