from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEG2_PER_SKY = 41252.96124941927
C_KM_S = 299792.458


@dataclass(frozen=True)
class SurveyBin:
    """
    One spectroscopic survey redshift bin.

    Parameters
    ----------
    z_min, z_max : float
        Redshift-bin edges.
    nbar_h3_mpc3 : float
        Comoving number density in h^3/Mpc^3.
    bias : float
        Linear galaxy bias used for P_g = b^2 P_m.
    label : str
        Human-readable bin/tracer label.
    kmax_h_mpc : float or None
        Optional per-bin maximum k in h/Mpc.
    """
    z_min: float
    z_max: float
    nbar_h3_mpc3: float
    bias: float = 1.5
    label: str = ""
    kmax_h_mpc: float | None = None

    @property
    def z_mid(self):
        return 0.5 * (self.z_min + self.z_max)

    def contains(self, z):
        z = float(z)
        return self.z_min <= z < self.z_max


@dataclass(frozen=True)
class SpectroscopicSurveySpec:
    """
    Minimal spectroscopic galaxy-clustering survey specification.
    """
    name: str
    area_deg2: float
    bins: tuple[SurveyBin, ...]
    notes: str = ""

    @property
    def f_sky(self):
        return self.area_deg2 / DEG2_PER_SKY


def e_z(z, omega_m=0.315, omega_l=0.685):
    """
    Dimensionless flat-LCDM expansion rate E(z).
    """
    z = np.asarray(z, dtype=float)
    return np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)


def comoving_distance_mpc(
    z,
    *,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
    n_grid=4096,
):
    """
    Flat-LCDM line-of-sight comoving distance in Mpc.

    This avoids adding an astropy dependency. It is accurate enough for
    survey-noise prototype; to be replaced with CAMB background
    distances.
    """
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    z_flat = np.atleast_1d(z)

    out = np.zeros_like(z_flat, dtype=float)

    for i, zi in enumerate(z_flat):
        if zi <= 0.0:
            out[i] = 0.0
            continue

        grid = np.linspace(0.0, zi, n_grid)
        integral = np.trapezoid(1.0 / e_z(grid, omega_m=omega_m, omega_l=omega_l), grid)
        out[i] = (C_KM_S / H0) * integral

    return out[0] if scalar else out.reshape(z.shape)


def comoving_distance_mpc_over_h(
    z,
    *,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Comoving distance in Mpc/h.
    """
    return h * comoving_distance_mpc(
        z,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )


def survey_bin_volume_h3_mpc3(
    survey,
    survey_bin,
    *,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Comoving shell volume for one survey bin in (Mpc/h)^3.
    """
    r_min = comoving_distance_mpc_over_h(
        survey_bin.z_min,
        h=h,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )
    r_max = comoving_distance_mpc_over_h(
        survey_bin.z_max,
        h=h,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )

    return (4.0 * np.pi / 3.0) * survey.f_sky * (r_max**3 - r_min**3)


def find_survey_bin(survey, z):
    """
    Return the survey bin containing redshift z.

    The upper edge is treated inclusively for the final bin.
    """
    z = float(z)

    for i, b in enumerate(survey.bins):
        if b.z_min <= z < b.z_max:
            return b

        if i == len(survey.bins) - 1 and np.isclose(z, b.z_max):
            return b

    raise ValueError(
        f"z={z} is outside survey {survey.name!r} redshift coverage."
    )


def estimate_delta_k(k):
    """
    Estimate per-point k-bin widths from a k-centre array.
    """
    k = np.asarray(k, dtype=float)

    if k.ndim != 1:
        raise ValueError("k must be one-dimensional.")

    if k.size < 2:
        raise ValueError("Need at least two k values to estimate delta-k.")

    edges = np.empty(k.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (k[:-1] + k[1:])
    edges[0] = k[0] - 0.5 * (k[1] - k[0])
    edges[-1] = k[-1] + 0.5 * (k[-1] - k[-2])

    edges[0] = max(edges[0], 0.0)

    return np.diff(edges)


def number_of_modes(
    k,
    delta_k,
    volume_h3_mpc3,
):
    """
    Number of independent Fourier modes in a spherical k-bin.

    N_modes = V * 4 pi k^2 delta_k / (2 pi)^3.
    """
    k = np.asarray(k, dtype=float)
    delta_k = np.asarray(delta_k, dtype=float)

    nmodes = volume_h3_mpc3 * 4.0 * np.pi * k**2 * delta_k / (2.0 * np.pi) ** 3

    return np.maximum(nmodes, 1.0)


def galaxy_power_sigma(
    k,
    p_matter,
    *,
    nbar_h3_mpc3,
    bias,
    volume_h3_mpc3,
    delta_k=None,
):
    """
    Gaussian diagonal uncertainty for isotropic galaxy P(k).

    sigma_Pg = sqrt(2 / N_modes) * [P_g + 1/nbar],
    where P_g = b^2 P_m.
    """
    k = np.asarray(k, dtype=float)
    p_matter = np.asarray(p_matter, dtype=float)

    if delta_k is None:
        delta_k = estimate_delta_k(k)
    else:
        delta_k = np.asarray(delta_k, dtype=float)

    p_galaxy = bias**2 * p_matter
    nmodes = number_of_modes(k, delta_k, volume_h3_mpc3)

    return np.sqrt(2.0 / nmodes) * (p_galaxy + 1.0 / nbar_h3_mpc3)


def ratio_observable_sigma(
    k,
    p_van_nl_matter,
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
    Diagonal uncertainty for R(k,z)=P_wig,nl/P_van,nl.

    Assume the observed tracer is P_g=b^2 P_m and propagate the Gaussian
    galaxy-power error to the ratio by dividing by b^2 P_van,nl.
    """
    k = np.asarray(k, dtype=float)
    p_van_nl_matter = np.asarray(p_van_nl_matter, dtype=float)

    b = find_survey_bin(survey, z)
    volume = survey_bin_volume_h3_mpc3(
        survey,
        b,
        h=h,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )

    sigma_pg = galaxy_power_sigma(
        k,
        p_van_nl_matter,
        nbar_h3_mpc3=b.nbar_h3_mpc3,
        bias=b.bias,
        volume_h3_mpc3=volume,
        delta_k=delta_k,
    )

    denom = b.bias**2 * p_van_nl_matter

    return sigma_pg / np.maximum(np.abs(denom), 1e-300)


def matter_power_observable_sigma(
    k,
    p_van_nl_matter,
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
    Diagonal uncertainty on an inferred matter P(k), converted from galaxy P(k).

    This is sigma_Pg / b^2.
    """
    k = np.asarray(k, dtype=float)
    p_van_nl_matter = np.asarray(p_van_nl_matter, dtype=float)

    b = find_survey_bin(survey, z)
    volume = survey_bin_volume_h3_mpc3(
        survey,
        b,
        h=h,
        H0=H0,
        omega_m=omega_m,
        omega_l=omega_l,
    )

    sigma_pg = galaxy_power_sigma(
        k,
        p_van_nl_matter,
        nbar_h3_mpc3=b.nbar_h3_mpc3,
        bias=b.bias,
        volume_h3_mpc3=volume,
        delta_k=delta_k,
    )

    return sigma_pg / b.bias**2


def survey_sigma_for_forecast_block(
    block,
    *,
    survey,
    observable="ratio_nl",
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Survey uncertainty for one forecast block returned by forecasting.py.
    """
    if observable == "ratio_nl":
        return ratio_observable_sigma(
            block["k"],
            block["p_van_nl"],
            survey=survey,
            z=block["z"],
            h=h,
            H0=H0,
            omega_m=omega_m,
            omega_l=omega_l,
        )

    if observable == "p_wig_nl":
        return matter_power_observable_sigma(
            block["k"],
            block["p_van_nl"],
            survey=survey,
            z=block["z"],
            h=h,
            H0=H0,
            omega_m=omega_m,
            omega_l=omega_l,
        )

    raise ValueError("observable must be either 'ratio_nl' or 'p_wig_nl'.")


def survey_sigma_for_forecast_vector(
    forecast,
    *,
    survey,
    h=0.677276,
    H0=67.7276,
    omega_m=0.315,
    omega_l=0.685,
):
    """
    Survey uncertainty for a stacked forecast data vector.
    """
    observable = forecast["observable"]

    sigmas = [
        survey_sigma_for_forecast_block(
            block,
            survey=survey,
            observable=observable,
            h=h,
            H0=H0,
            omega_m=omega_m,
            omega_l=omega_l,
        )
        for block in forecast["blocks"]
    ]

    return np.concatenate(sigmas)


def combine_independent_sigmas(*sigmas):
    """
    Combine independent diagonal standard deviations by inverse variance.
    """
    arrays = [np.asarray(s, dtype=float) for s in sigmas if s is not None]

    if len(arrays) == 0:
        raise ValueError("At least one sigma array is required.")

    shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError("All sigma arrays must have the same shape.")

    invvar = np.zeros(shape, dtype=float)

    for arr in arrays:
        invvar += 1.0 / np.maximum(arr, 1e-300) ** 2

    return 1.0 / np.sqrt(invvar)


def _surface_density_to_bin_nbar(
    *,
    area_deg2,
    z_edges,
    total_surface_density_deg2,
    bias_model,
    name_prefix,
):
    """
    Convert a total surface density over a redshift range into uniform
    per-bin comoving densities.

    Convenient fallback when literature gives total surface
    densities but not a tabulated n(z) in the exact format needed.
    """
    temp_survey = SpectroscopicSurveySpec(
        name="temporary",
        area_deg2=area_deg2,
        bins=tuple(),
    )

    z_edges = np.asarray(z_edges, dtype=float)
    total_surface_density_deg2 = float(total_surface_density_deg2)

    total_number = total_surface_density_deg2 * area_deg2

    # Distribute uniformly in redshift for now. This is a placeholder until
    # survey-specific n(z) tables are inserted.
    n_bins = len(z_edges) - 1
    number_per_bin = total_number / n_bins

    bins = []

    for i in range(n_bins):
        z_min = float(z_edges[i])
        z_max = float(z_edges[i + 1])
        z_mid = 0.5 * (z_min + z_max)

        b_tmp = SurveyBin(
            z_min=z_min,
            z_max=z_max,
            nbar_h3_mpc3=1.0,
        )

        volume = survey_bin_volume_h3_mpc3(temp_survey, b_tmp)
        nbar = number_per_bin / volume

        bins.append(
            SurveyBin(
                z_min=z_min,
                z_max=z_max,
                nbar_h3_mpc3=nbar,
                bias=float(bias_model(z_mid)),
                label=f"{name_prefix}_{i}",
            )
        )

    return tuple(bins)


def desi_like_survey(
    *,
    extended=False,
):
    """
    DESI-like spectroscopic survey spec.

    This is a broad effective forecast spec, not a final official DESI n(z)
    table. It uses literature-level target/sample scales and should be
    replaced by exact DESI n(z), bias tables for publication-quality results.
    """
    area = 17000.0 if extended else 14000.0

    # Broad effective bins with rough tracer-motivated surface-density scales.
    # BGS: >10M over 14k deg2 at z<0.6.
    # LRG: ~8M over 14k deg2 at 0.4<z<1.1.
    # ELG: target selection scale ~2387 deg^-2, here reduced to a conservative
    # effective spectroscopic density.
    # QSO: target density ~310 deg^-2, shot-noise limited.
    bins = []

    bins += list(
        _surface_density_to_bin_nbar(
            area_deg2=area,
            z_edges=[0.1, 0.3, 0.5, 0.7],
            total_surface_density_deg2=10_000_000.0 / 14000.0,
            bias_model=lambda z: 1.2 + 0.8 * z,
            name_prefix="DESI_BGS",
        )
    )

    bins += list(
        _surface_density_to_bin_nbar(
            area_deg2=area,
            z_edges=[0.7, 0.9, 1.1],
            total_surface_density_deg2=8_000_000.0 / 14000.0,
            bias_model=lambda z: 1.7 + 0.7 * (z - 0.7),
            name_prefix="DESI_LRG",
        )
    )

    bins += list(
        _surface_density_to_bin_nbar(
            area_deg2=area,
            z_edges=[1.1, 1.3, 1.5, 1.7],
            total_surface_density_deg2=1200.0,
            bias_model=lambda z: 1.0 + 0.84 * z,
            name_prefix="DESI_ELG",
        )
    )

    bins += list(
        _surface_density_to_bin_nbar(
            area_deg2=area,
            z_edges=[1.7, 2.0, 2.3],
            total_surface_density_deg2=310.0,
            bias_model=lambda z: 2.1 + 0.3 * (z - 1.7),
            name_prefix="DESI_QSO",
        )
    )

    return SpectroscopicSurveySpec(
        name="DESI-like extended" if extended else "DESI-like",
        area_deg2=area,
        bins=tuple(bins),
        notes=(
            "Broad effective DESI-like spec. Replace with exact DESI n(z), "
            "bias and completeness tables for publication-quality forecasts."
        ),
    )


def euclid_feature_paper_survey():
    """
    Euclid GCsp survey specification matched to the Euclid primordial-features
    forecast setup.

    Source:
        Euclid Collaboration, "Euclid: The search for primordial features",
        arXiv:2309.17287.

    The GCsp setup uses four H-alpha spectroscopic bins centred at
        z = {1.0, 1.2, 1.4, 1.65},
    with bin widths
        Delta z = {0.2, 0.2, 0.2, 0.3}.

    The galaxy number densities are
        n(z) = {6.86, 5.58, 4.21, 2.61}e-4 h^3/Mpc^3,

    and the H-alpha galaxy biases are
        b(z) = {1.46, 1.61, 1.75, 1.90}.

    This is the Euclid survey specification used for the proof-of-concept
    primordial-feature forecast.
    """
    return SpectroscopicSurveySpec(
        name="Euclid GCsp primordial-features spec",
        area_deg2=15000.0,
        bins=(
            SurveyBin(
                z_min=0.9,
                z_max=1.1,
                nbar_h3_mpc3=6.86e-4,
                bias=1.46,
                label="EUCLID_GCsp_z1p0",
            ),
            SurveyBin(
                z_min=1.1,
                z_max=1.3,
                nbar_h3_mpc3=5.58e-4,
                bias=1.61,
                label="EUCLID_GCsp_z1p2",
            ),
            SurveyBin(
                z_min=1.3,
                z_max=1.5,
                nbar_h3_mpc3=4.21e-4,
                bias=1.75,
                label="EUCLID_GCsp_z1p4",
            ),
            SurveyBin(
                z_min=1.5,
                z_max=1.8,
                nbar_h3_mpc3=2.61e-4,
                bias=1.90,
                label="EUCLID_GCsp_z1p65",
            ),
        ),
        notes=(
            "Matched to Euclid Collaboration, arXiv:2309.17287: "
            "four GCsp H-alpha bins, tabulated n(z), and tabulated bias."
        ),
    )


def roman_hlss_survey():
    """
    Roman HLSS-like spectroscopic survey.

    Uses 2000 deg^2, roughly 10M H-alpha redshifts at 1<z<2 and 2M [OIII]
    redshifts at 2<z<3.
    """
    area = 2000.0

    ha_bins = _surface_density_to_bin_nbar(
        area_deg2=area,
        z_edges=np.arange(1.0, 2.0 + 1e-12, 0.1),
        total_surface_density_deg2=10_000_000.0 / 2000.0,
        bias_model=lambda z: 0.9 + 0.4 * z,
        name_prefix="ROMAN_HA",
    )

    oiii_bins = _surface_density_to_bin_nbar(
        area_deg2=area,
        z_edges=np.arange(2.0, 3.0 + 1e-12, 0.2),
        total_surface_density_deg2=2_000_000.0 / 2000.0,
        bias_model=lambda z: 1.2 + 0.45 * z,
        name_prefix="ROMAN_OIII",
    )

    return SpectroscopicSurveySpec(
        name="Roman HLSS-like",
        area_deg2=area,
        bins=tuple(list(ha_bins) + list(oiii_bins)),
        notes=(
            "Roman HLSS-like spec from reference-area and redshift-count "
            "figures. Replace with exact Roman n(z), bias tables when inserted."
        ),
    )


def print_survey_summary(survey):
    """
    Print a compact survey summary for sanity checks.
    """
    print(f"\n{survey.name}")
    print("-" * len(survey.name))
    print(f"area_deg2: {survey.area_deg2:g}")
    print(f"f_sky:     {survey.f_sky:.5f}")
    if survey.notes:
        print(f"notes:     {survey.notes}")

    print("\nBins")
    print("----")
    for b in survey.bins:
        print(
            f"{b.label:14s} "
            f"z=[{b.z_min:.2f}, {b.z_max:.2f}], "
            f"z_mid={b.z_mid:.2f}, "
            f"nbar={b.nbar_h3_mpc3:.4e} h^3/Mpc^3, "
            f"bias={b.bias:.3f}"
        )