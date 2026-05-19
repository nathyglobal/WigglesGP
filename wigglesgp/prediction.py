from __future__ import annotations

from typing import Literal

import numpy as np

from .damping import feature_delta, gaussian_damping


Observable = Literal["ratio", "power"]


def build_prediction_block(
    *,
    emulator,
    redshift,
    log10omega,
    A_feat,
    phi,
    k,
    p_reference,
    feature_type="log",
    k_fit_min=0.05,
    k_fit_max=0.6,
    h=0.67,
    check_domain=True,
    propagate_gp_uncertainty=True,
    apply_damping=True,
):
    """
    Build one redshift block for the feature prediction.

    The internal modelling object is the relative feature contribution,

        delta_pred(k,z) = D(k,z) * A cos[omega f(k) + phi],

    where D=1 for the undamped comparison and D is the GP-calibrated damping
    envelope for the damped comparison.

    The two supported observable representations are

        ratio(k,z) = 1 + delta_pred(k,z),

    and

        power(k,z) = P_ref(k,z) * ratio(k,z),

    where P_ref is the featureless linear matter power spectrum at the fixed
    fiducial cosmology.
    """
    k = np.asarray(k, dtype=float)
    p_reference = np.asarray(p_reference, dtype=float)

    if k.shape != p_reference.shape:
        raise ValueError(
            f"k and p_reference shapes differ: {k.shape} vs {p_reference.shape}."
        )

    mask = (k >= float(k_fit_min)) & (k <= float(k_fit_max))

    if not np.any(mask):
        raise ValueError(f"No k values inside [{k_fit_min}, {k_fit_max}].")

    k = k[mask]
    p_reference = p_reference[mask]

    z_grid = np.full_like(k, float(redshift), dtype=float)
    omega_grid = np.full_like(k, float(log10omega), dtype=float)

    delta_lin = feature_delta(
        k,
        log10omega,
        phi,
        feature_type=feature_type,
        amplitude=A_feat,
        h=h,
    )

    if apply_damping:
        if propagate_gp_uncertainty:
            sigma, sigma_std = emulator.predict_sigma(
                z_grid,
                omega_grid,
                return_std=True,
                check_domain=check_domain,
            )
            damping = gaussian_damping(k, sigma, h=h)
        else:
            damping, sigma = emulator.damping(
                k,
                z_grid,
                omega_grid,
                return_sigma=True,
                check_domain=check_domain,
            )
            sigma_std = np.zeros_like(sigma, dtype=float)
    else:
        sigma = np.zeros_like(k, dtype=float)
        sigma_std = np.zeros_like(k, dtype=float)
        damping = np.ones_like(k, dtype=float)

    delta_pred = damping * delta_lin

    ratio = 1.0 + delta_pred
    power = p_reference * ratio

    # ratio = 1 + D delta_lin, with D = exp[-(h k Sigma)^2 / 2].
    # Therefore
    #
    #   d ratio / d Sigma = -delta_pred * (h k)^2 Sigma.
    #
    # This propagates only the GP predictive uncertainty in Sigma.
    ratio_gp_std = np.abs(delta_pred * (h * k) ** 2 * sigma * sigma_std)
    power_gp_std = np.abs(p_reference) * ratio_gp_std

    return {
        "z": float(redshift),
        "log10omega": float(log10omega),
        "A_feat": float(A_feat),
        "phi": float(phi),
        "k": k,
        "p_reference": p_reference,
        "delta_lin": delta_lin,
        "damping": damping,
        "sigma": sigma,
        "sigma_std": sigma_std,
        "delta_pred": delta_pred,
        "ratio": ratio,
        "power": power,
        "ratio_gp_std": ratio_gp_std,
        "power_gp_std": power_gp_std,
    }


def build_prediction_vector(
    *,
    emulator,
    baseline_cache,
    redshifts,
    log10omega,
    A_feat,
    phi,
    feature_type="log",
    k_fit_min=0.05,
    k_fit_max=0.6,
    k_fit_max_by_z=None,
    h=0.67,
    check_domain=True,
    observable: Observable = "power",
    propagate_gp_uncertainty=True,
    apply_damping=True,
):
    """
    Build a stacked prediction vector over redshift blocks.

    Supported observables are:

        ratio : dimensionless feature ratio, P_pred / P_ref
        power : matter-power bandpower, P_ref * ratio
    """
    observable_key = _normalise_observable(observable)

    redshifts = [float(z) for z in redshifts]
    blocks = []

    for redshift in redshifts:
        if redshift not in baseline_cache:
            raise KeyError(f"No baseline spectrum cached for z={redshift}.")

        baseline = baseline_cache[redshift]

        if k_fit_max_by_z is None:
            k_fit_max_this_z = k_fit_max
        else:
            k_fit_max_this_z = float(k_fit_max_by_z.get(float(redshift), k_fit_max))

        if "p_reference" not in baseline:
            raise KeyError("Baseline block must contain 'p_reference'.")

        block = build_prediction_block(
            emulator=emulator,
            redshift=redshift,
            log10omega=log10omega,
            A_feat=A_feat,
            phi=phi,
            k=baseline["k"],
            p_reference=baseline["p_reference"],
            feature_type=feature_type,
            k_fit_min=k_fit_min,
            k_fit_max=k_fit_max_this_z,
            h=h,
            check_domain=check_domain,
            propagate_gp_uncertainty=propagate_gp_uncertainty,
            apply_damping=apply_damping,
        )

        blocks.append(block)

    vector = np.concatenate([block[observable_key] for block in blocks])

    if observable_key == "ratio":
        gp_std = np.concatenate([block["ratio_gp_std"] for block in blocks])
    elif observable_key == "power":
        gp_std = np.concatenate([block["power_gp_std"] for block in blocks])
    else:
        raise RuntimeError("Unexpected observable key.")

    k = np.concatenate([block["k"] for block in blocks])
    z = np.concatenate(
        [
            np.full_like(block["k"], block["z"], dtype=float)
            for block in blocks
        ]
    )

    return {
        "vector": vector,
        "k": k,
        "z": z,
        "blocks": blocks,
        "observable": observable_key,
        "gp_std": gp_std,
    }


def _normalise_observable(observable: str) -> Observable:
    if observable == "ratio":
        return "ratio"

    if observable == "power":
        return "power"

    raise ValueError("observable must be either 'ratio' or 'power'.")


def total_diagonal_sigma(
    vector,
    *,
    observational_sigma=None,
    gp_sigma=None,
    model_error_sigma=None,
    floor=1e-12,
):
    """
    Combine independent diagonal standard deviations in quadrature.
    """
    vector = np.asarray(vector, dtype=float)
    variance = np.zeros_like(vector, dtype=float)

    if observational_sigma is not None:
        observational_sigma = np.asarray(observational_sigma, dtype=float)
        if observational_sigma.shape != vector.shape:
            raise ValueError(
                f"observational_sigma shape {observational_sigma.shape} "
                f"does not match vector shape {vector.shape}."
            )
        variance += observational_sigma**2

    if gp_sigma is not None:
        gp_sigma = np.asarray(gp_sigma, dtype=float)
        if gp_sigma.shape != vector.shape:
            raise ValueError(
                f"gp_sigma shape {gp_sigma.shape} "
                f"does not match vector shape {vector.shape}."
            )
        variance += gp_sigma**2

    if model_error_sigma is not None:
        model_error_sigma = np.asarray(model_error_sigma, dtype=float)
        if model_error_sigma.shape != vector.shape:
            raise ValueError(
                f"model_error_sigma shape {model_error_sigma.shape} "
                f"does not match vector shape {vector.shape}."
            )
        variance += model_error_sigma**2

    return np.maximum(np.sqrt(variance), floor)


def gaussian_loglike_diagonal(model, data, sigma):
    """
    Gaussian diagonal log-likelihood, excluding normalisation constants.
    """
    model = np.asarray(model, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if model.shape != data.shape or model.shape != sigma.shape:
        raise ValueError(
            f"Shape mismatch: model={model.shape}, data={data.shape}, "
            f"sigma={sigma.shape}."
        )

    resid = (model - data) / sigma
    chi2 = float(np.sum(resid**2))

    return -0.5 * chi2, chi2