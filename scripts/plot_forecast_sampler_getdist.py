#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np


DEFAULTS = {
    "log": {
        "input": Path("forecast_tests/BOBE_log_forecast_non_fiducial_sampler_euclid_2d.npz"),
        "output": Path("forecast_tests/BOBE_log_forecast_non_fiducial_sampler_getdist_euclid_2d.pdf"),
        "title": "Log feature forecast, Euclid GCsp sampler",
    },
    "linear": {
        "input": Path("forecast_tests/BOBE_linear_forecast_non_fiducial_sampler_euclid_2d.npz"),
        "output": Path("forecast_tests/BOBE_linear_forecast_non_fiducial_sampler_getdist_euclid_2d.pdf"),
        "title": "Linear feature forecast, Euclid GCsp sampler",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot dynesty forecast sampler output with GetDist."
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
    )

    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument(
        "--phase-scale",
        choices=["rad", "twopi", "pi"],
        default="twopi",
    )

    parser.add_argument("--title", default=None)

    return parser.parse_args()


def resolve(args):
    d = DEFAULTS[args.feature_type]
    return (
        args.input or d["input"],
        args.output or d["output"],
        args.title if args.title is not None else d["title"],
    )


def phase_column(phi, phase_scale):
    if phase_scale == "rad":
        return phi, r"\phi"
    if phase_scale == "pi":
        return phi / np.pi, r"\phi/\pi"
    if phase_scale == "twopi":
        return phi / (2.0 * np.pi), r"\phi/(2\pi)"
    raise ValueError(phase_scale)


def main():
    args = parse_args()
    input_path, output_path, title = resolve(args)

    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    data = np.load(input_path, allow_pickle=True)

    samples_raw = np.asarray(data["samples"], dtype=float)
    weights = np.asarray(data["weights"], dtype=float)

    omega = samples_raw[:, 0]
    phi, phi_label = phase_column(samples_raw[:, 1], args.phase_scale)

    samples = np.column_stack([omega, phi])

    gd = MCSamples(
        samples=samples,
        weights=weights,
        names=["omega_label", "phi"],
        labels=[r"\omega_{\rm label}", phi_label],
        name_tag="Forecast",
    )

    gd.updateSettings(
        {
            "contours": [0.68, 0.95],
            "smooth_scale_2D": 0.6,
            "smooth_scale_1D": 0.6,
        }
    )

    print("\nSampler summary")
    print("---------------")
    print(f"input:       {input_path}")
    print(f"n samples:   {samples.shape[0]}")
    print(f"ESS approx:  {(weights.sum() ** 2 / np.sum(weights ** 2)):.1f}")
    print(f"logZ:        {data['logz'][-1]:.6g} +/- {data['logzerr'][-1]:.6g}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = plots.get_subplot_plotter(width_inch=7.0)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14

    g.triangle_plot(
        [gd],
        ["omega_label", "phi"],
        filled=True,
    )

    if title:
        plt.suptitle(title, fontsize=14, y=0.98)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()