#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np


DEFAULTS = {
    "log": {
        "input": Path("forecast_tests/log_forecast_grid_local_getdist_Afixed_101x101_gp_model.csv"),
        "output": Path("forecast_tests/log_forecast_getdist_corner_101x101_gp_model.pdf"),
        "title": "Log feature forecast",
    },
    "linear": {
        "input": Path("forecast_tests/linear_forecast_grid_local_getdist_Afixed_101x101_gp_model.csv"),
        "output": Path("forecast_tests/linear_forecast_getdist_corner_101x101_gp_model.pdf"),
        "title": "Linear feature forecast",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a forecast likelihood grid into weighted GetDist samples "
            "and make a corner plot."
        )
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature template. Used to set default input/output paths.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input forecast grid CSV. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output corner plot path. Defaults from --feature-type.",
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
        help="Optional plot title. Defaults from --feature-type.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    defaults = DEFAULTS[args.feature_type]

    input_path = args.input or defaults["input"]
    output_path = args.output or defaults["output"]
    title = args.title if args.title is not None else defaults["title"]

    return input_path, output_path, title


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


def make_getdist_samples(
    data,
    *,
    include_A=False,
    phase_scale="twopi",
    weight_floor=1e-300,
    thin_zero_weight=False,
):
    from getdist import MCSamples

    A = np.asarray(data["A_feat"], dtype=float)
    omega_label = np.asarray(data["log10omega"], dtype=float)
    phi_plot, phi_label = phase_column(data["phi"], phase_scale)

    delta_chi2 = np.asarray(data["delta_chi2"], dtype=float)
    delta_chi2 = delta_chi2 - np.nanmin(delta_chi2)

    weights = np.exp(-0.5 * delta_chi2)

    if thin_zero_weight:
        mask = np.isfinite(weights) & (weights > weight_floor)
    else:
        weights = np.maximum(weights, weight_floor)
        mask = np.isfinite(weights)

    candidate_columns = []
    candidate_names = []
    candidate_labels = []

    if include_A:
        candidate_columns.append(A)
        candidate_names.append("A_feat")
        candidate_labels.append(r"A_{\rm feat}")

    candidate_columns.append(omega_label)
    candidate_names.append("omega_label")
    candidate_labels.append(r"\omega_{\rm label}")

    candidate_columns.append(phi_plot)
    candidate_names.append("phi")
    candidate_labels.append(phi_label)

    columns = []
    names = []
    labels = []

    for col, name, label in zip(candidate_columns, candidate_names, candidate_labels):
        col_masked = np.asarray(col[mask], dtype=float)

        if col_masked.size == 0:
            continue

        span = np.nanmax(col_masked) - np.nanmin(col_masked)

        if not np.isfinite(span) or span <= 0.0:
            print(f"Skipping constant parameter in GetDist plot: {name}")
            continue

        columns.append(col_masked)
        names.append(name)
        labels.append(label)

    if len(columns) == 0:
        raise ValueError("No varying parameters available for GetDist plot.")

    samples = np.column_stack(columns)
    weights = weights[mask]

    if samples.shape[0] == 0:
        raise ValueError("No finite-weight samples survived.")

    ranges = {
        name: (float(np.nanmin(col)), float(np.nanmax(col)))
        for name, col in zip(names, columns)
    }

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        names=names,
        labels=labels,
        ranges=ranges,
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
    print(f"best omega_label:  {data['log10omega'][best]:.6g}")
    print(f"best omega_model:  {10.0 ** data['log10omega'][best]:.6g}")
    print(f"best phi:          {data['phi'][best]:.6g}")


def main():
    args = parse_args()
    input_path, output_path, title = resolve_defaults(args)

    import matplotlib.pyplot as plt
    from getdist import plots

    data = load_grid(input_path)
    print_summary(data)

    samples = make_getdist_samples(
        data,
        include_A=args.include_A,
        phase_scale=args.phase_scale,
        weight_floor=args.weight_floor,
        thin_zero_weight=args.thin_zero_weight,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = plots.get_subplot_plotter(width_inch=7.0)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.settings.legend_fontsize = 12
    g.settings.figure_legend_frame = False

    params = [p.name for p in samples.getParamNames().names]

    g.triangle_plot(
        [samples],
        params,
        filled=True,
        title_limit=1,
    )

    if title is not None:
        plt.suptitle(title, fontsize=14, y=0.98)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()