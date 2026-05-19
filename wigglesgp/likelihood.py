"""
Likelihood for comparing information in damped, undamped and linear-only
wiggle predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .binning import top_hat_bin_average_prediction
from .camb_power import build_baseline_spectra_cache
from .emulator import SigmaEmulator
from .prediction import (
    build_prediction_vector,
    gaussian_loglike_diagonal,
    total_diagonal_sigma,
)
from .survey import euclid_feature_paper_survey, survey_sigma_for_prediction_vector


Scenario = Literal["full_damped", "full_undamped", "linear_only"]
FeatureType = Literal["log", "linear"]
Observable = Literal["ratio", "power"]


@dataclass(frozen=True)
class WiggleComparisonConfig:
    feature_type: FeatureType
    emulator_path: Path

    scenario: Scenario = "full_damped"
    observable: Observable = "ratio"
    redshifts: tuple[float, ...] = (1.0, 1.2, 1.4, 1.65)

    fid_A_feat: float = 0.03
    fid_omega: float = 1.26
    fid_phi: float = np.pi

    omega_min: float = 0.8
    omega_max: float = 2.0
    A_min: float = 0.0
    A_max: float = 0.06
    phi_min: float = 0.0
    phi_max: float = 2.0 * np.pi

    kmax: float = 0.8
    k_fit_min: float = 0.05
    k_fit_max: float = 0.6
    internal_npoints: int = 2000

    bin_delta_k: float = 0.004
    keep_partial_final_bin: bool = False

    include_gp_uncertainty: bool = False
    model_error_floor: float = 5e-3

    check_domain: bool = True
    h: float = 0.67
    linear_cut_n_omega: int = 256


def damping_threshold_kmax_by_z(
    *,
    emulator,
    redshifts,
    omega_min,
    omega_max,
    amplitude=0.03,
    epsilon0=5e-3,
    h=0.67,
    n_omega=256,
    k_fit_max=0.6,
    check_domain=True,
):
    """
    Redshift-dependent linear-only cutoff from the calibrated damping emulator.

    The cutoff is the largest k for which the maximum damping-induced change in
    the feature contribution remains below epsilon0:

        A [1 - D(k,z,Sigma_max)] < epsilon0.
    """
    if epsilon0 <= 0:
        raise ValueError("epsilon0 must be positive.")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive.")
    if epsilon0 >= amplitude:
        raise ValueError("epsilon0 must be smaller than amplitude.")

    omega_grid = np.linspace(float(omega_min), float(omega_max), int(n_omega))
    damping_min = 1.0 - float(epsilon0) / float(amplitude)
    prefactor = np.sqrt(-2.0 * np.log(damping_min))

    kmax_by_z = {}
    for z in redshifts:
        z_grid = np.full_like(omega_grid, float(z), dtype=float)
        sigma_grid = emulator.predict_sigma(
            z_grid,
            omega_grid,
            return_std=False,
            check_domain=check_domain,
        )
        sigma_max = float(np.max(sigma_grid))
        k_cut = prefactor / (float(h) * sigma_max)
        kmax_by_z[float(z)] = min(float(k_cut), float(k_fit_max))

    return kmax_by_z


class WiggleComparisonLikelihood:
    """
    Gaussian diagonal likelihood for the wiggle information comparison.

    The Asimov data vector is generated from the same prediction pipeline at
    the injected fiducial feature parameters. The sampled parameter order is

        theta = [A_feat, log10omega, phi].
    """

    def __init__(self, config: WiggleComparisonConfig):
        self.config = config

        if not config.emulator_path.exists():
            raise FileNotFoundError(f"Emulator file does not exist: {config.emulator_path}")

        self.emulator = SigmaEmulator.from_file(config.emulator_path)
        self.survey = euclid_feature_paper_survey()

        self.apply_damping, self.k_fit_max_by_z = self._resolve_scenario()
        self.internal_k_fit_min, self.internal_k_fit_max = self._internal_k_range()
        self.internal_k_fit_max_by_z = self._internal_kmax_by_z()

        self.baseline_cache = build_baseline_spectra_cache(
            redshifts=config.redshifts,
            kmax=config.kmax,
            npoints=config.internal_npoints,
        )

        self.fid_fine = self._build_prediction(
            A_feat=config.fid_A_feat,
            omega_label=config.fid_omega,
            phi=config.fid_phi,
            propagate_gp_uncertainty=config.include_gp_uncertainty,
        )
        self.fid = self._maybe_bin(self.fid_fine)
        self.data = self.fid["vector"]

        self.survey_sigma = survey_sigma_for_prediction_vector(self.fid, survey=self.survey)

        model_error_sigma = self._model_error_sigma(self.fid)

        self.sigma = total_diagonal_sigma(
            self.data,
            observational_sigma=self.survey_sigma,
            gp_sigma=self.fid["gp_std"] if config.include_gp_uncertainty else None,
            model_error_sigma=model_error_sigma,
        )

        self.eval_counter = 0

    @property
    def ndim(self):
        return 3

    @property
    def param_list(self):
        return ["A_feat", "omega_label", "phi"]

    @property
    def param_bounds(self):
        c = self.config
        return np.array(
            [
                [c.A_min, c.A_max],
                [c.omega_min, c.omega_max],
                [c.phi_min, c.phi_max],
            ],
            dtype=float,
        ).T

    @property
    def param_labels(self):
        return [r"\mathcal{A}", r"\log_{10}(\omega)", r"\phi"]

    def _resolve_scenario(self):
        c = self.config
        linear_kmax_by_z = damping_threshold_kmax_by_z(
            emulator=self.emulator,
            redshifts=c.redshifts,
            omega_min=c.omega_min,
            omega_max=c.omega_max,
            amplitude=c.fid_A_feat,
            epsilon0=c.model_error_floor,
            h=c.h,
            n_omega=c.linear_cut_n_omega,
            k_fit_max=c.k_fit_max,
            check_domain=c.check_domain,
        )

        if c.feature_type == "log":
            if c.scenario == "full_damped":
                return True, None
            if c.scenario == "full_undamped":
                return False, None
            if c.scenario == "linear_only":
                return True, linear_kmax_by_z
            raise ValueError(f"Unknown scenario: {c.scenario}")

        # Scenario comparison is currently only used for logarithmic features.
        return True, None

    def _internal_k_range(self):
        c = self.config
        pad = c.bin_delta_k if c.bin_delta_k > 0.0 else 0.0
        return max(1e-5, c.k_fit_min - pad), min(c.kmax, c.k_fit_max + pad)

    def _internal_kmax_by_z(self):
        c = self.config
        if self.k_fit_max_by_z is not None and c.bin_delta_k > 0.0:
            return {
                float(z): min(float(kmax_z) + c.bin_delta_k, c.kmax)
                for z, kmax_z in self.k_fit_max_by_z.items()
            }
        return self.k_fit_max_by_z
    
    def _model_error_sigma(self, prediction):
        eps = float(self.config.model_error_floor)

        if eps <= 0.0:
            return None
        
        if prediction["observable"] == "ratio":
            return np.full_like(prediction["vector"], eps, dtype=float)
        
        if prediction["observable"] == "power":
            p_ref = np.concatenate(
                [
                    np.asarray(block["p_reference"], dtype=float) 
                    for block in prediction["blocks"]
                ]
            )
            return eps * np.abs(p_ref)
        
        raise ValueError("observable must be either 'ratio' or 'power'.")

    def _build_prediction(self, *, A_feat, omega_label, phi, propagate_gp_uncertainty):
        c = self.config
        return build_prediction_vector(
            emulator=self.emulator,
            baseline_cache=self.baseline_cache,
            redshifts=c.redshifts,
            log10omega=omega_label,
            A_feat=A_feat,
            phi=phi,
            feature_type=c.feature_type,
            k_fit_min=self.internal_k_fit_min,
            k_fit_max=self.internal_k_fit_max,
            k_fit_max_by_z=self.internal_k_fit_max_by_z,
            h=c.h,
            check_domain=c.check_domain,
            observable=c.observable,
            propagate_gp_uncertainty=propagate_gp_uncertainty,
            apply_damping=self.apply_damping,
        )

    def _maybe_bin(self, prediction):
        c = self.config
        if c.bin_delta_k <= 0.0:
            return prediction
        return top_hat_bin_average_prediction(
            prediction,
            k_min=c.k_fit_min,
            k_max=c.k_fit_max,
            delta_k=c.bin_delta_k,
            k_max_by_z=self.k_fit_max_by_z,
            drop_partial_final_bin=not c.keep_partial_final_bin,
        )

    def model_prediction(self, theta):
        A_feat, omega_label, phi = theta
        fine = self._build_prediction(
            A_feat=A_feat,
            omega_label=omega_label,
            phi=phi,
            propagate_gp_uncertainty=False,
        )
        return self._maybe_bin(fine)

    def model_vector(self, theta):
        return self.model_prediction(theta)["vector"]

    def loglike(self, theta):
        self.eval_counter += 1
        ll, _ = gaussian_loglike_diagonal(self.model_vector(theta), self.data, self.sigma)
        return ll

    def chi2(self, theta):
        _, chi2 = gaussian_loglike_diagonal(self.model_vector(theta), self.data, self.sigma)
        return chi2

    def summary(self):
        c = self.config
        lines = [
            "WiggleComparisonLikelihood",
            "--------------------------",
            f"feature_type:           {c.feature_type}",
            f"scenario:               {c.scenario}",
            f"observable:             {self.fid['observable']}",
            f"apply_damping:          {self.apply_damping}",
            f"redshifts:              {c.redshifts}",
            f"fid_A_feat:             {c.fid_A_feat}",
            f"fid_omega:              {c.fid_omega}",
            f"fid_phi:                {c.fid_phi}",
            f"A prior:                [{c.A_min}, {c.A_max}]",
            f"omega prior:            [{c.omega_min}, {c.omega_max}]",
            f"phi prior:              [{c.phi_min}, {c.phi_max}]",
            f"k_fit range:            [{c.k_fit_min}, {c.k_fit_max}]",
            f"internal k range:       [{self.internal_k_fit_min}, {self.internal_k_fit_max}]",
            f"bin_delta_k:            {c.bin_delta_k}",
            f"keep_partial_final_bin: {c.keep_partial_final_bin}",
            f"k_fit_max_by_z:         {self.k_fit_max_by_z}",
            f"internal_kmax_by_z:     {self.internal_k_fit_max_by_z}",
            f"data vector length:     {self.data.size}",
            (
                "survey sigma min/med/max: "
                f"{np.nanmin(self.survey_sigma):.6g} / "
                f"{np.nanmedian(self.survey_sigma):.6g} / "
                f"{np.nanmax(self.survey_sigma):.6g}"
            ),
            (
                "total sigma min/med/max:  "
                f"{np.nanmin(self.sigma):.6g} / "
                f"{np.nanmedian(self.sigma):.6g} / "
                f"{np.nanmax(self.sigma):.6g}"
            ),
        ]

        for block in self.fid["blocks"]:
            dk = block.get("delta_k", np.diff(block["k"]))
            lines.append(
                f"z={block['z']}: Nk={block['k'].size}, "
                f"k=[{block['k'].min():.4f}, {block['k'].max():.4f}], "
                f"dk min/med/max={np.min(dk):.4g}/{np.median(dk):.4g}/{np.max(dk):.4g}"
            )
        return "\n".join(lines)
