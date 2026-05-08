#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


DEFAULTS = {
    "log": {
        "input": Path("forecast_tests/log_forecast_grid_local_getdist_Afixed_101x101_gp_model.csv"),
        "output": Path("forecast_tests/log_forecast_grid_slices.pdf"),
        "fixed_omega": 1.26,
        "fixed_phi": np.pi,
    },
    "linear": {
        "input": Path("forecast_tests/linear_forecast_grid_local_getdist_Afixed_101x101_gp_model.csv"),
        "output": Path("forecast_tests/linear_forecast_grid_slices.pdf"),
        "fixed_omega": 0.87,
        "fixed_phi": np.pi,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot forecast likelihood grid slices."
    )

    parser.add_argument(
        "--feature-type",
        choices=["log", "linear"],
        required=True,
        help="Feature template. Used only to set sensible default paths/labels.",
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
        help="Output PDF path. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--fixed-A",
        type=float,
        default=0.03,
        help="A value used for the omega-phi slice.",
    )

    parser.add_argument(
        "--fixed-omega",
        type=float,
        default=None,
        help="Omega-label value used for the A-phi slice. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--fixed-phi",
        type=float,
        default=None,
        help="Phi value used for the A-omega slice. Defaults from --feature-type.",
    )

    parser.add_argument(
        "--max-delta-chi2",
        type=float,
        default=None,
        help="Optional maximum Delta chi2 shown in colour scale.",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=40,
        help="Number of filled-contour levels.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    defaults = DEFAULTS[args.feature_type]

    input_path = args.input or defaults["input"]
    output_path = args.output or defaults["output"]
    fixed_omega = args.fixed_omega if args.fixed_omega is not None else defaults["fixed_omega"]
    fixed_phi = args.fixed_phi if args.fixed_phi is not None else defaults["fixed_phi"]

    return input_path, output_path, fixed_omega, fixed_phi


def set_latex_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.unicode_minus": False,
        }
    )


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


def nearest_value(values, target):
    values = np.asarray(values, dtype=float)
    unique = np.unique(values)
    idx = np.argmin(np.abs(unique - target))
    return float(unique[idx])


def make_slice(data, fixed_name, fixed_value):
    values = data[fixed_name]
    actual = nearest_value(values, fixed_value)
    mask = np.isclose(values, actual, rtol=0.0, atol=1e-12)
    return data[mask], actual


def plot_tricontour(ax, x, y, z, xlabel, ylabel, title, vmax=None, levels=40):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[finite]
    y = y[finite]
    z = z[finite]

    if len(x) < 3:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Not enough points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        return None

    z = z - np.nanmin(z)

    if vmax is not None:
        z_plot = np.minimum(z, vmax)
        level_values = np.linspace(np.nanmin(z_plot), vmax, levels)
    else:
        z_plot = z
        level_values = levels

    triang = mtri.Triangulation(x, y)

    im = ax.tricontourf(
        triang,
        z_plot,
        levels=level_values,
    )

    for level in [2.30, 6.18, 11.83]:
        if np.nanmin(z) <= level <= np.nanmax(z):
            ax.tricontour(
                triang,
                z,
                levels=[level],
                colors="black",
                linewidths=0.8,
            )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=17, pad=8)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=13,
    )

    return im


def main():
    args = parse_args()
    input_path, output_path, fixed_omega, fixed_phi = resolve_defaults(args)

    set_latex_style()

    data = load_grid(input_path)

    A_unique = np.unique(data["A_feat"])
    omega_unique = np.unique(data["log10omega"])
    phi_unique = np.unique(data["phi"])

    panels = []

    if len(omega_unique) > 1 and len(phi_unique) > 1:
        sl, actual_A = make_slice(data, "A_feat", args.fixed_A)
        panels.append(
            {
                "data": sl,
                "x": "log10omega",
                "y": "phi",
                "xlabel": r"$\omega_{\rm label}$",
                "ylabel": r"$\phi$",
                "title": rf"$A_{{\rm feat}}={actual_A:g}$",
            }
        )

    if len(A_unique) > 1 and len(omega_unique) > 1:
        sl, actual_phi = make_slice(data, "phi", fixed_phi)
        panels.append(
            {
                "data": sl,
                "x": "log10omega",
                "y": "A_feat",
                "xlabel": r"$\omega_{\rm label}$",
                "ylabel": r"$A_{\rm feat}$",
                "title": rf"$\phi={actual_phi:g}$",
            }
        )

    if len(A_unique) > 1 and len(phi_unique) > 1:
        sl, actual_omega = make_slice(data, "log10omega", fixed_omega)
        panels.append(
            {
                "data": sl,
                "x": "phi",
                "y": "A_feat",
                "xlabel": r"$\phi$",
                "ylabel": r"$A_{\rm feat}$",
                "title": rf"$\omega_{{\rm label}}={actual_omega:g}$",
            }
        )

    if not panels:
        raise ValueError(
            "No two-dimensional slices are available. "
            "Use at least two varying parameters in the grid."
        )

    n_panels = len(panels)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.2 * n_panels, 5.2),
        squeeze=False,
    )

    axes = axes[0]
    images = []

    for ax, panel in zip(axes, panels):
        sl = panel["data"]

        im = plot_tricontour(
            ax,
            sl[panel["x"]],
            sl[panel["y"]],
            sl["delta_chi2"],
            panel["xlabel"],
            panel["ylabel"],
            panel["title"],
            vmax=args.max_delta_chi2,
            levels=args.levels,
        )

        if im is not None:
            images.append(im)

    if images:
        cbar = fig.colorbar(
            images[-1],
            ax=axes,
            fraction=0.035,
            pad=0.025,
        )
        cbar.set_label(r"$\Delta\chi^2$", fontsize=16)
        cbar.ax.tick_params(labelsize=13)

    fig.subplots_adjust(
        left=0.065,
        right=0.92,
        bottom=0.13,
        top=0.88,
        wspace=0.30,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()