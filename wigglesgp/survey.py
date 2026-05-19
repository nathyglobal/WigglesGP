"""
Euclid-like spectroscopic bandpower covariance used in the wiggle information
comparison.

This is not a full Euclid GCsp observable model. It supplies the redshift bins,
number densities, galaxy biases, survey volume and diagonal Gaussian bandpower
uncertainties used for the relative comparison of damping scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEG2_PER_SKY = 41252.96124941927
C_KM_S = 299792.458


@dataclass(frozen=True)
class SurveyBin:
    z_min: float
    z_max: float
    nbar_h3_mpc3: float
    bias: float
    label: str = ""

    @property
    def z_mid(self):
        return 0.5 * (self.z_min + self.z_max)


@dataclass(frozen=True)
class SpectroscopicSurveySpec:
    name: str
    area_deg2: float
    bins: tuple[SurveyBin, ...]
    notes: str = ""

    @property
    def f_sky(self):
        return self.area_deg2 / DEG2_PER_SKY


def euclid_feature_paper_survey():
    """
    Euclid GCsp-like H-alpha sample used for the primordial-feature comparison.

    Bins are centred at z={1.0, 1.2, 1.4, 1.65} with widths
    {0.2, 0.2, 0.2, 0.3}. Number densities are in h^3/Mpc^3.
    """
    return SpectroscopicSurveySpec(
        name="Euclid GCsp primordial-features spec",
        area_deg2=15000.0,
        bins=(
            SurveyBin(0.9, 1.1, 6.86e-4, 1.46, "EUCLID_GCsp_z1p0"),
            SurveyBin(1.1, 1.3, 5.58e-4, 1.61, "EUCLID_GCsp_z1p2"),
            SurveyBin(1.3, 1.5, 4.21e-4, 1.75, "EUCLID_GCsp_z1p4"),
            SurveyBin(1.5, 1.8, 2.61e-4, 1.90, "EUCLID_GCsp_z1p65"),
        ),
        notes=(
            "Four Euclid GCsp H-alpha bins used for the primordial-feature "
            "information comparison. This is a simplified 3D bandpower "
            "covariance, not the full Euclid likelihood."
        ),
    )


def e_z(z, omega_m=0.315, omega_l=0.685):
    z = np.asarray(z, dtype=float)
    return np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)


def comoving_distance_mpc(z, *, H0=67.7276, omega_m=0.315, omega_l=0.685, n_grid=4096):
    """Flat-LCDM line-of-sight comoving distance in Mpc."""
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    z_flat = np.atleast_1d(z)
    out = np.zeros_like(z_flat, dtype=float)

    for i, zi in enumerate(z_flat):
        if zi <= 0.0:
            out[i] = 0.0
            continue
        grid = np.linspace(0.0, zi, n_grid)
        out[i] = (C_KM_S / H0) * np.trapezoid(1.0 / e_z(grid, omega_m, omega_l), grid)

    return out[0] if scalar else out.reshape(z.shape)


def comoving_distance_mpc_over_h(z, *, h=0.677276, H0=67.7276, omega_m=0.315, omega_l=0.685):
    return h * comoving_distance_mpc(z, H0=H0, omega_m=omega_m, omega_l=omega_l)


def survey_bin_volume_h3_mpc3(survey, survey_bin, *, h=0.677276, H0=67.7276, omega_m=0.315, omega_l=0.685):
    """Comoving shell volume for one survey bin in (Mpc/h)^3."""
    r_min = comoving_distance_mpc_over_h(survey_bin.z_min, h=h, H0=H0, omega_m=omega_m, omega_l=omega_l)
    r_max = comoving_distance_mpc_over_h(survey_bin.z_max, h=h, H0=H0, omega_m=omega_m, omega_l=omega_l)
    return (4.0 * np.pi / 3.0) * survey.f_sky * (r_max**3 - r_min**3)


def find_survey_bin(survey, z):
    z = float(z)
    for i, bin_ in enumerate(survey.bins):
        if bin_.z_min <= z < bin_.z_max:
            return bin_
        if i == len(survey.bins) - 1 and np.isclose(z, bin_.z_max):
            return bin_
    raise ValueError(f"z={z} is outside survey {survey.name!r} coverage.")


def estimate_delta_k(k):
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size < 2:
        raise ValueError("k must be one-dimensional with at least two values.")

    edges = np.empty(k.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (k[:-1] + k[1:])
    edges[0] = max(k[0] - 0.5 * (k[1] - k[0]), 0.0)
    edges[-1] = k[-1] + 0.5 * (k[-1] - k[-2])
    return np.diff(edges)

def gaussian_mode_count(k, delta_k, volume_h3_mpc3):
    """
    Number of Fourier modes in an isotropic spherical k-shell.

    N_modes = V * 4 pi k^2 Delta k / (2 pi)^3.

    This is the simplified isotropic analogue of the GCsp Fisher weighting.
    The full Euclid GCsp treatment integrates over both k and mu and uses
    an effective volume. Here we retain only the spherical-bandpower mode count.
    """
    k = np.asarray(k, dtype=float)
    delta_k = np.asarray(delta_k, dtype=float)

    nmodes = volume_h3_mpc3 * 4.0 * np.pi * k**2 * delta_k / (2.0 * np.pi) ** 3
    return np.maximum(nmodes, 1.0)


def galaxy_bandpower_sigma(
    k,
    p_reference_matter,
    *,
    nbar_h3_mpc3,
    bias,
    volume_h3_mpc3,
    delta_k=None,
):
    """
    Diagonal Gaussian uncertainty for simplified 3D galaxy-power bandpowers.

    We approximate the observed spectroscopic galaxy power as

        P_g(k,z) = b^2(z) P_ref(k,z)

    and use

        sigma[P_g] = sqrt(2 / N_modes) * [P_g + 1/nbar].

    This keeps the survey covariance in the same 3D power-spectrum units as
    the Euclid GCsp setup, but omits the full mu dependence, RSD, AP factors,
    redshift errors, FoG damping, and off-diagonal window-function covariance.
    """
    k = np.asarray(k, dtype=float)
    p_reference_matter = np.asarray(p_reference_matter, dtype=float)

    if delta_k is None:
        delta_k = estimate_delta_k(k)
    else:
        delta_k = np.asarray(delta_k, dtype=float)

    p_galaxy = bias**2 * p_reference_matter
    nmodes = gaussian_mode_count(k, delta_k, volume_h3_mpc3)

    return np.sqrt(2.0 / nmodes) * (p_galaxy + 1.0 / nbar_h3_mpc3)


def matter_bandpower_sigma(
    k,
    p_reference,
    *,
    survey,
    z,
    delta_k=None,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Diagonal uncertainty for the inferred matter-power bandpower.

    The survey noise is first written in galaxy-power units and then converted
    back to matter-power units by dividing by b^2:

        sigma[P_m] = sigma[P_g] / b^2.
    """
    k = np.asarray(k, dtype=float)
    p_reference = np.asarray(p_reference, dtype=float)

    bin_ = find_survey_bin(survey, z)

    volume = survey_bin_volume_h3_mpc3(
        survey,
        bin_,
        h=h,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )

    sigma_pg = galaxy_bandpower_sigma(
        k,
        p_reference,
        nbar_h3_mpc3=bin_.nbar_h3_mpc3,
        bias=bin_.bias,
        volume_h3_mpc3=volume,
        delta_k=delta_k,
    )

    return sigma_pg / bin_.bias**2


def ratio_bandpower_sigma(
    k,
    p_reference,
    *,
    survey,
    z,
    delta_k=None,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Diagonal uncertainty for the ratio observable

        R(k,z) = P_pred(k,z) / P_ref(k,z).

    This is not an independent survey observable. It is the same simplified
    3D matter-power covariance transformed into ratio units:

        sigma[R] = sigma[P_m] / P_ref.
    """
    p_reference = np.asarray(p_reference, dtype=float)

    sigma_pm = matter_bandpower_sigma(
        k,
        p_reference,
        survey=survey,
        z=z,
        delta_k=delta_k,
        h=h,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )

    return sigma_pm / np.maximum(np.abs(p_reference), 1e-300)


def survey_sigma_for_prediction_block(
    block,
    *,
    survey,
    observable,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Survey uncertainty for one prediction block.

    For observable="power", the returned sigma has units of the matter power
    spectrum.

    For observable="ratio", the same matter-power covariance is divided by the
    reference matter power spectrum so that it is expressed in dimensionless
    ratio units.
    """
    delta_k = block.get("delta_k", None)

    if "p_reference" not in block:
        raise KeyError("Prediction block must contain 'p_reference'.")

    p_reference = np.asarray(block["p_reference"], dtype=float)

    if observable == "power":
        return matter_bandpower_sigma(
            block["k"],
            p_reference,
            survey=survey,
            z=block["z"],
            delta_k=delta_k,
            h=h,
            H0=H0,
            omega_m=omega_m,
            omega_l=omega_l,
        )

    if observable == "ratio":
        return ratio_bandpower_sigma(
            block["k"],
            p_reference,
            survey=survey,
            z=block["z"],
            delta_k=delta_k,
            h=h,
            H0=H0,
            omega_m=omega_m,
            omega_l=omega_l,
        )

    raise ValueError("observable must be either 'ratio' or 'power'.")


def survey_sigma_for_prediction_vector(
    prediction,
    *,
    survey,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Survey uncertainty for a stacked prediction vector.
    """
    observable = prediction["observable"]

    return np.concatenate(
        [
            survey_sigma_for_prediction_block(
                block,
                survey=survey,
                observable=observable,
                h=h,
                H0=H0,
                omega_m=omega_m,
                omega_l=omega_l,
            )
            for block in prediction["blocks"]
        ]
    )

def print_survey_summary(survey):
    print(f"\n{survey.name}")
    print("-" * len(survey.name))
    print(f"area_deg2: {survey.area_deg2:g}")
    print(f"f_sky:     {survey.f_sky:.5f}")
    if survey.notes:
        print(f"notes:     {survey.notes}")
    print("\nBins")
    print("----")
    for bin_ in survey.bins:
        print(
            f"{bin_.label:14s} z=[{bin_.z_min:.2f}, {bin_.z_max:.2f}], "
            f"z_mid={bin_.z_mid:.2f}, nbar={bin_.nbar_h3_mpc3:.4e} h^3/Mpc^3, "
            f"bias={bin_.bias:.3f}"
        )
