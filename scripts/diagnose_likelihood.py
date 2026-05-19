from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from wigglesgp.likelihood import WiggleComparisonConfig, WiggleComparisonLikelihood


def plot_data_vs_fiducial_model(like, output="forecast_tests/diagnostic_data_vs_model.pdf"):
    c = like.config
    theta_fid = np.array([c.fid_A_feat, c.fid_omega, c.fid_phi])
    model_prediction = like.model_prediction(theta_fid)

    fig, axes = plt.subplots(
        len(like.fid["blocks"]),
        1,
        figsize=(8, 2.2 * len(like.fid["blocks"])),
        sharex=True,
        constrained_layout=True,
    )

    if len(like.fid["blocks"]) == 1:
        axes = [axes]

    for ax, data_block, model_block in zip(axes, like.fid["blocks"], model_prediction["blocks"]):
        k = data_block["k"]
        z = data_block["z"]

        y_data = data_block[like.fid["observable"]]
        y_model = model_block[model_prediction["observable"]]

        ax.plot(k, y_data, marker="o", markersize=2.2, linewidth=1.0, label="Asimov data")
        ax.plot(k, y_model, marker="x", markersize=2.2, linewidth=0.8, label="Fiducial model")
        ax.axhline(1.0, linewidth=0.8, alpha=0.5)

        ax.set_ylabel(r"$R(k,z)$")
        ax.text(
            0.02,
            0.95,
            rf"$z={z:g}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )

    axes[-1].set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    axes[0].legend(frameon=False)
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


def plot_1d_likelihood_slices(like, output="forecast_tests/diagnostic_1d_slices.pdf"):
    c = like.config
    theta_fid = np.array([c.fid_A_feat, c.fid_omega, c.fid_phi])
    chi2_fid = like.chi2(theta_fid)

    grids = [
        (r"$\mathcal{A}$", 0, np.linspace(c.A_min, c.A_max, 160)),
        (r"$\log_{10}\omega$", 1, np.linspace(c.omega_min, c.omega_max, 180)),
        (r"$\phi$", 2, np.linspace(c.phi_min, c.phi_max, 180)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)

    for ax, (label, idx, grid) in zip(axes, grids):
        dchi2 = []

        for value in grid:
            theta = theta_fid.copy()
            theta[idx] = value
            dchi2.append(like.chi2(theta) - chi2_fid)

        dchi2 = np.asarray(dchi2)

        ax.plot(grid, dchi2, linewidth=1.2)
        ax.axvline(theta_fid[idx], linestyle="--", linewidth=1.0)
        ax.axhline(1.0, linestyle=":", linewidth=0.9)
        ax.axhline(4.0, linestyle=":", linewidth=0.9)

        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta\chi^2$")

    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


def plot_omega_phi_surface(
    like,
    output="forecast_tests/diagnostic_omega_phi_surface.pdf",
    n_omega=90,
    n_phi=90,
):
    c = like.config
    theta_fid = np.array([c.fid_A_feat, c.fid_omega, c.fid_phi])
    chi2_fid = like.chi2(theta_fid)

    omega_grid = np.linspace(c.omega_min, c.omega_max, n_omega)
    phi_grid = np.linspace(c.phi_min, c.phi_max, n_phi)

    dchi2 = np.empty((n_phi, n_omega), dtype=float)

    for i, phi in enumerate(phi_grid):
        for j, omega in enumerate(omega_grid):
            theta = np.array([c.fid_A_feat, omega, phi])
            dchi2[i, j] = like.chi2(theta) - chi2_fid

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)

    vmax = np.nanpercentile(dchi2, 95)
    im = ax.pcolormesh(
        omega_grid,
        phi_grid,
        dchi2,
        shading="auto",
        vmin=0.0,
        vmax=vmax,
    )

    levels = [2.30, 6.18, 11.83]
    ax.contour(
        omega_grid,
        phi_grid,
        dchi2,
        levels=levels,
        linewidths=1.0,
    )

    ax.scatter(
        [c.fid_omega],
        [c.fid_phi],
        marker="*",
        s=110,
        edgecolor="black",
        zorder=5,
        label="Fiducial",
    )

    ax.set_xlabel(r"$\log_{10}\omega$")
    ax.set_ylabel(r"$\phi$")
    ax.legend(frameon=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta\chi^2$")

    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    config = WiggleComparisonConfig(
        feature_type="log",
        emulator_path=Path("emulators/log_sigma_gp.pkl"),
        scenario="full_undamped",
        observable="ratio",
        fid_A_feat=0.03,
        fid_omega=1.26,
        fid_phi=np.pi,
        omega_min=0.8,
        omega_max=2.0,
        bin_delta_k=0.004,
        internal_npoints=2000,
        keep_partial_final_bin=False,
    )

    like = WiggleComparisonLikelihood(config)

    print(like.summary())

    theta_fid = np.array([config.fid_A_feat, config.fid_omega, config.fid_phi])
    print("chi2(fid) =", like.chi2(theta_fid))

    plot_data_vs_fiducial_model(like)
    plot_1d_likelihood_slices(like)
    plot_omega_phi_surface(like)