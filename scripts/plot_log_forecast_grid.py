#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot forecast likelihood grid slices."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("forecast_tests/log_forecast_grid.csv"),
        help="Input forecast grid CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forecast_tests/log_forecast_grid_slices.pdf"),
        help="Output PDF path.",
    )

    parser.add_argument(
        "--fixed-A",
        type=float,
        default=0.03,
        help="A value used for the log10omega-phi slice.",
    )

    parser.add_argument(
        "--fixed-log10omega",
        type=float,
        default=1.26,
        help="log10omega value used for the A-phi slice.",
    )

    parser.add_argument(
        "--fixed-phi",
        type=float,
        default=0.0,
        help="phi value used for the A-log10omega slice.",
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

    # Useful reference contours for Gaussian-style confidence regions.
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
    set_latex_style()

    data = load_grid(args.input)

    A_unique = np.unique(data["A_feat"])
    omega_unique = np.unique(data["log10omega"])
    phi_unique = np.unique(data["phi"])

    panels = []

    # Panel 1: log10omega-phi at fixed A
    if len(omega_unique) > 1 and len(phi_unique) > 1:
        sl, actual_A = make_slice(data, "A_feat", args.fixed_A)
        panels.append(
            {
                "data": sl,
                "x": "log10omega",
                "y": "phi",
                "xlabel": r"$\log_{10}\omega$",
                "ylabel": r"$\phi$",
                "title": rf"$A_{{\rm feat}}={actual_A:g}$",
            }
        )

    # Panel 2: A-log10omega at fixed phi
    if len(A_unique) > 1 and len(omega_unique) > 1:
        sl, actual_phi = make_slice(data, "phi", args.fixed_phi)
        panels.append(
            {
                "data": sl,
                "x": "log10omega",
                "y": "A_feat",
                "xlabel": r"$\log_{10}\omega$",
                "ylabel": r"$A_{\rm feat}$",
                "title": rf"$\phi={actual_phi:g}$",
            }
        )

    # Panel 3: A-phi at fixed log10omega
    if len(A_unique) > 1 and len(phi_unique) > 1:
        sl, actual_omega = make_slice(data, "log10omega", args.fixed_log10omega)
        panels.append(
            {
                "data": sl,
                "x": "phi",
                "y": "A_feat",
                "xlabel": r"$\phi$",
                "ylabel": r"$A_{\rm feat}$",
                "title": rf"$\log_{{10}}\omega={actual_omega:g}$",
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()