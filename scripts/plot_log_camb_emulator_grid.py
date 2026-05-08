#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot CAMB + Sigma-emulator forecast-grid outputs. "
            "Each panel compares the linear feature ratio to the "
            "emulator-damped non-linear feature ratio."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("forecast_tests/log_camb_emulator_grid"),
        help="Directory containing per-grid-point CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forecast_tests/log_camb_emulator_grid_matrix.pdf"),
        help="Output plot path.",
    )

    parser.add_argument(
        "--feature-type",
        default="log",
        help="Feature type prefix used in filenames.",
    )

    parser.add_argument(
        "--z-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 5.0],
        help="Redshifts to plot, in row order.",
    )

    parser.add_argument(
        "--log10omega-values",
        type=float,
        nargs="+",
        default=[0.8, 1.26, 2.0],
        help="Frequency labels to plot, in column order.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[13.0, 14.0],
        help="Figure size as width height.",
    )

    return parser.parse_args()


def format_float_for_filename(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def format_float_label(value):
    return f"{float(value):g}"


def load_grid_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    required = [
        "k",
        "ratio_lin",
        "ratio_nl_emulator",
        "damping",
        "sigma",
    ]

    for name in required:
        if name not in data.dtype.names:
            raise ValueError(
                f"Missing column {name!r} in {path}. "
                f"Available columns are {data.dtype.names}."
            )

    return data


def set_latex_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.unicode_minus": False,
        }
    )


def main():
    args = parse_args()
    set_latex_style()

    z_values = [float(z) for z in args.z_values]
    omega_values = [float(w) for w in args.log10omega_values]

    n_z = len(z_values)
    n_w = len(omega_values)

    fig, axes = plt.subplots(
        n_z,
        n_w,
        figsize=tuple(args.figsize),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    all_ratios = []

    for i, z in enumerate(z_values):
        for j, omega in enumerate(omega_values):
            ax = axes[i, j]

            z_label = format_float_for_filename(z)
            omega_label = format_float_for_filename(omega)

            path = args.input_dir / f"{args.feature_type}_z{z_label}_w{omega_label}.csv"

            if not path.exists():
                ax.set_axis_off()
                print(f"Missing file: {path}")
                continue

            data = load_grid_csv(path)

            k = data["k"]
            ratio_lin = data["ratio_lin"]
            ratio_nl = data["ratio_nl_emulator"]
            damping = data["damping"]
            sigma = data["sigma"]

            all_ratios.append(ratio_lin[np.isfinite(ratio_lin)])
            all_ratios.append(ratio_nl[np.isfinite(ratio_nl)])

            ax.plot(
                k,
                ratio_lin,
                color="tab:blue",
                linestyle="--",
                linewidth=1.2,
                label=r"$R_{\rm lin}$" if (i == 0 and j == 0) else None,
            )

            ax.plot(
                k,
                ratio_nl,
                color="tab:orange",
                linestyle="-",
                linewidth=1.3,
                label=r"$R_{\rm emu}$" if (i == 0 and j == 0) else None,
            )

            ax.axhline(
                1.0,
                color="black",
                linestyle=":",
                linewidth=0.8,
            )

            ax.text(
                0.04,
                0.08,
                rf"$\Sigma={np.nanmedian(sigma):.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=11,
            )

            if i == 0:
                ax.set_title(
                    rf"$\log_{{10}}\omega={format_float_label(omega)}$",
                    fontsize=16,
                    pad=8,
                )

            if j == 0:
                ax.set_ylabel(
                    rf"$z={format_float_label(z)}$" + "\n" + r"$R(k)$",
                    fontsize=15,
                )

            if i == n_z - 1:
                ax.set_xlabel(r"$k\,[h/{\rm Mpc}]$", fontsize=15)

            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=True,
                labelsize=12,
                length=3,
            )

            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    if all_ratios:
        finite = np.concatenate(all_ratios)
        finite = finite[np.isfinite(finite)]

        if finite.size > 0:
            ymin = np.nanmin(finite)
            ymax = np.nanmax(finite)

            pad = 0.08 * (ymax - ymin)
            ymin -= pad
            ymax += pad

            # Keep the visual scale centred enough around unity.
            ymin = min(ymin, 0.965)
            ymax = max(ymax, 1.035)

            for ax in axes.ravel():
                if ax.has_data():
                    ax.set_ylim(ymin, ymax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=15,
            bbox_to_anchor=(0.5, 1.005),
        )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.06,
        top=0.94,
        wspace=0.0,
        hspace=0.0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()