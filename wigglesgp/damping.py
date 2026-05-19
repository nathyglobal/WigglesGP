from __future__ import annotations

import numpy as np


FeatureType = str


def feature_delta(
    k,
    log10omega,
    phi,
    *,
    feature_type: FeatureType,
    amplitude=0.03,
    k_pivot=0.05,
    h=0.67,
):
    """
    Analytic linear fractional feature contribution.

    Returns
        delta_lin(k) = A cos[omega f(k) + phi]

    where omega = 10**log10omega. For logarithmic oscillations,
    f(k)=ln(k/k_pivot). For linear oscillations, f(k)=k/k_pivot.
    The input k is in h/Mpc and k_pivot is in Mpc^-1, so h*k is used in
    the argument.
    """
    k = np.asarray(k, dtype=float)
    omega = 10.0 ** np.asarray(log10omega, dtype=float)
    k_mpc = h * k

    if np.any(k_mpc <= 0.0):
        raise ValueError("All k values must be positive for feature templates.")

    if feature_type == "log":
        argument = omega * np.log(k_mpc / float(k_pivot)) + phi
    elif feature_type == "linear":
        argument = omega * (k_mpc / float(k_pivot)) + phi
    else:
        raise ValueError("feature_type must be either 'log' or 'linear'.")

    return float(amplitude) * np.cos(argument)


def gaussian_damping(k, sigma, *, h=0.67):
    """
    Gaussian damping envelope D(k,Sigma)=exp[-(h k Sigma)^2/2].

    k is in h/Mpc and Sigma is in Mpc, following the calibration convention.
    """
    k = np.asarray(k, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.exp(-0.5 * (float(h) * k * sigma) ** 2)


def damped_feature_delta(
    k,
    log10omega,
    phi,
    sigma,
    *,
    feature_type: FeatureType,
    amplitude=0.03,
    k_pivot=0.05,
    h=0.67,
):
    """
    Damped fractional feature contribution  delta_lin(k) * D(k,Sigma).
    """
    return gaussian_damping(k, sigma, h=h) * feature_delta(
        k,
        log10omega,
        phi,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )


def feature_ratio(
    k,
    log10omega,
    phi,
    sigma=0.0,
    *,
    feature_type: FeatureType,
    amplitude=0.03,
    k_pivot=0.05,
    h=0.67,
):
    """
    Multiplicative feature ratio 1 + delta_lin(k) * D(k,Sigma).
    """
    return 1.0 + damped_feature_delta(
        k,
        log10omega,
        phi,
        sigma,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )