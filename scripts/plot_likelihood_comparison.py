#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


SCENARIO_LABELS = {
    "full_damped": r"Full $k$ range, $\mathcal{P}_{\mathrm{m,pred}}^{\mathrm{damped}}$",
    "full_undamped": r"Full $k$ range, $\mathcal{P}_{\mathrm{m,pred}}^{\mathrm{undamped}}$",
    "linear_only": r"Linear $k$ only",
}

SCENARIO_ORDER = ["full_damped", "full_undamped", "linear_only"]


def weighted_quantile(x, q, weights=None):
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)

    if weights is None:
        return np.quantile(x, q)

    weights = np.asarray(weights, dtype=float)
    sorter = np.argsort(x)
    x = x[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(q, cdf, x)


def scenario_from_file(path):
    d = np.load(path, allow_pickle=True)
    if "scenario" in d:
        return str(d["scenario"])
    name = Path(path).stem
    for scenario in SCENARIO_ORDER:
        if scenario in name:
            return scenario
    raise ValueError(f"Could not infer scenario from {path}")


def load_getdist_sample(path, label):
    d = np.load(path, allow_pickle=True)

    samples = np.asarray(d["samples"], dtype=float)
    weights = np.asarray(d["weights"], dtype=float)

    samples_plot = samples.copy()
    samples_plot[:, 2] = samples_plot[:, 2] / (2.0 * np.pi)

    names = ["A_feat", "omega_label", "phi_over_2pi"]
    labels = [r"\mathcal{A}", r"\log_{10}(\omega)", r"\phi/2\pi"]

    gd = MCSamples(
        samples=samples_plot,
        weights=weights,
        names=names,
        labels=labels,
        label=label,
    )

    fid = {
        "A_feat": float(d["fid_A_feat"]),
        "omega_label": float(d["fid_omega"]),
        "phi_over_2pi": float(d["fid_phi"] / (2.0 * np.pi)),
    }

    return gd, fid, d


def compute_summary(path):
    d = np.load(path, allow_pickle=True)

    samples = np.asarray(d["samples"], dtype=float)
    weights = np.asarray(d["weights"], dtype=float)

    vals = {
        "A_feat": samples[:, 0],
        "omega_label": samples[:, 1],
        "phi_over_2pi": samples[:, 2] / (2.0 * np.pi),
    }

    out = {}

    for key, x in vals.items():
        q16, q50, q84 = weighted_quantile(x, [0.16, 0.50, 0.84], weights)
        out[key] = {
            "median": q50,
            "upper": q84 - q50,
            "lower": q50 - q16,
        }

    return out


def format_pm(x):
    return f"{x['median']:.4f} +{x['upper']:.4f} -{x['lower']:.4f}"


def format_latex_pm(x):
    return (
        rf"${x['median']:.4f}^{{+{x['upper']:.4f}}}_"
        rf"{{-{x['lower']:.4f}}}$"
    )


def print_ascii_table(rows):
    headers = ["Set", "Scenario", "A", "log10(omega)", "phi/2pi"]

    widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]

    print("\nMarginal constraints")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    print(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))

    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def write_latex_table(rows, output):
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\hline")
    lines.append(r"Set & Scenario & $\mathcal{A}$ & $\log_{10}\omega$ & $\phi/2\pi$ \\")
    lines.append(r"\hline")

    for set_name, scenario_label, A, omega, phi in rows:
        lines.append(
            f"{set_name} & {scenario_label} & {A} & {omega} & {phi} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Marginal constraints from the three relative-information comparison scenarios.}"
    )
    lines.append(r"\label{tab:wiggle_comparison_constraints}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)

    print(f"\nWrote LaTeX table: {output}")


def make_triangle_plot(samples_list, fid, output):
    names = ["A_feat", "omega_label", "phi_over_2pi"]

    g = plots.get_subplot_plotter()

    g.triangle_plot(
        samples_list,
        names,
        filled=[True, False, False],
        diag1d=False,
        plot_1d=False,
        legend_loc="upper right",
    )

    # Hide diagonal 1D panels and upper triangle.
    for i in range(len(names)):
        ax = g.subplots[i, i]
        if ax is not None:
            ax.set_visible(False)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ax = g.subplots[i, j]
            if ax is not None:
                ax.set_visible(False)

    # Compact 2D-only triangle layout.
    pos_top_left = [0.12, 0.48, 0.36, 0.36]
    pos_bottom_left = [0.12, 0.12, 0.36, 0.36]
    pos_bottom_right = [0.48, 0.12, 0.36, 0.36]

    g.subplots[1, 0].set_position(pos_top_left)
    g.subplots[2, 0].set_position(pos_bottom_left)
    g.subplots[2, 1].set_position(pos_bottom_right)

    g.subplots[1, 0].set_xlabel("")
    g.subplots[1, 0].tick_params(labelbottom=False)

    g.subplots[2, 1].set_ylabel("")
    g.subplots[2, 1].tick_params(labelleft=False)

    for i in range(1, len(names)):
        for j in range(i):
            ax = g.subplots[i, j]

            x_name = names[j]
            y_name = names[i]

            x_fid = fid[x_name]
            y_fid = fid[y_name]

            ax.axvline(x_fid, linestyle="--", linewidth=1.1, color="black", alpha=0.9)
            ax.axhline(y_fid, linestyle="--", linewidth=1.1, color="black", alpha=0.9)
            ax.plot(
                x_fid,
                y_fid,
                color="black",
                marker="x",
                markersize=4,
                markeredgewidth=0.9,
                linestyle="None",
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    g.export(str(output))
    print(f"Wrote plot: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot three-scenario wiggle comparison results and write constraint tables."
    )

    parser.add_argument(
        "--set",
        nargs=4,
        action="append",
        metavar=("NAME", "FULL_DAMPED", "FULL_UNDAMPED", "LINEAR_ONLY"),
        required=True,
        help="One result set: name plus three scenario NPZ files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_tests"),
        help="Directory for plots and tables.",
    )

    parser.add_argument(
        "--plot-prefix",
        default="log_three_scenario_comparison",
        help="Prefix for plot/table output files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ascii_rows = []
    latex_rows = []

    for set_idx, set_args in enumerate(args.set):
        set_name = set_args[0]
        paths = {
            "full_damped": Path(set_args[1]),
            "full_undamped": Path(set_args[2]),
            "linear_only": Path(set_args[3]),
        }

        samples_list = []
        fid_ref = None

        for scenario in SCENARIO_ORDER:
            path = paths[scenario]
            label = SCENARIO_LABELS[scenario]

            gd, fid, _metadata = load_getdist_sample(path, label)

            if fid_ref is None:
                fid_ref = fid
            else:
                for key in fid_ref:
                    if not np.isclose(fid_ref[key], fid[key]):
                        raise ValueError(
                            f"Fiducial mismatch for {set_name}, {scenario}, {key}: "
                            f"{fid[key]} vs {fid_ref[key]}"
                        )

            samples_list.append(gd)

            summary = compute_summary(path)

            ascii_rows.append(
                [
                    set_name,
                    scenario,
                    format_pm(summary["A_feat"]),
                    format_pm(summary["omega_label"]),
                    format_pm(summary["phi_over_2pi"]),
                ]
            )

            latex_rows.append(
                [
                    set_name,
                    SCENARIO_LABELS[scenario],
                    format_latex_pm(summary["A_feat"]),
                    format_latex_pm(summary["omega_label"]),
                    format_latex_pm(summary["phi_over_2pi"]),
                ]
            )

        if len(args.set) == 1:
            plot_name = f"{args.plot_prefix}.pdf"
        else:
            safe_set_name = set_name.replace(" ", "_").replace("/", "_")
            plot_name = f"{args.plot_prefix}_{safe_set_name}.pdf"

        make_triangle_plot(
            samples_list,
            fid_ref,
            args.output_dir / plot_name,
        )

    print_ascii_table(ascii_rows)

    write_latex_table(
        latex_rows,
        args.output_dir / f"{args.plot_prefix}_constraints.tex",
    )


if __name__ == "__main__":
    main()