import numpy as np

from .damping import damped_wiggle_ratio
from .power import nonlinear_wiggle_power
from .camb_power import get_wiggle_linear_spectra_redshifts



def warn_if_uncalibrated_amplitude(A_feat, calibrated_A=0.03):
    """
    Warn when the feature amplitude differs from the value used to calibrate
    the Sigma emulator.

    The current Sigma emulator is calibrated only at fixed A_feat=0.03.
    Phase has been tested separately and does not enter the damping emulator,
    but amplitude dependence has not yet been calibrated.
    """
    if not np.isclose(float(A_feat), float(calibrated_A), rtol=0.0, atol=0.0):
        print(
            "WARNING: The Sigma emulator is calibrated only at "
            f"A_feat={calibrated_A:g}. "
            f"Received A_feat={float(A_feat):g}. "
            "Amplitude-dependence of the damping scale has not yet been "
            "calibrated, so this forecast assumes the same Sigma(z, omega) "
            "applies at this amplitude."
        )


def damped_feature_ratio_from_emulator(
    k,
    z,
    omega,
    emulator,
    feature_type,
    amplitude=0.03,
    phase=0.0,
    k_pivot=0.05,
    h=0.67,
    check_domain=True,
    return_components=False,
):
    """
    Build the emulator-damped feature ratio.

    Parameters
    ----------
    k : array-like
        Wavenumber values in h/Mpc.
    z : float or array-like
        Redshift values. Can be scalar or broadcastable to k.
    omega : float or array-like
        Stored frequency coordinate used by the Sigma emulator. In the current
        calibration this is the table/emulator omega label, i.e. log10(omega),
        not 10**omega.
    emulator : SigmaEmulator
        Loaded Sigma emulator.
    feature_type : {"log", "linear"}
        Feature template.
    amplitude : float
        Feature amplitude.
    phase : float
        Feature phase in radians.
    k_pivot : float
        Pivot scale used by the feature model.
    h : float
        Dimensionless Hubble parameter. The calibrated damping convention is
        exp[-(h k Sigma)^2 / 2].
    check_domain : bool
        If True, require z and omega to lie inside the emulator domain.
    return_components : bool
        If True, also return the undamped ratio, damping envelope, and Sigma.

    Returns
    -------
    ratio_damped : ndarray
        Emulator-damped feature ratio.
    components : dict, optional
        Returned only if return_components=True.
    """
    k = np.asarray(k, dtype=float)
    z = np.asarray(z, dtype=float)
    omega = np.asarray(omega, dtype=float)

    sigma = emulator.predict_sigma(
        z,
        omega,
        return_std=False,
        check_domain=check_domain,
    )

    damping = np.exp(-0.5 * (h * k * sigma) ** 2)

    omega_model = 10.0 ** omega

    ratio_linear = damped_wiggle_ratio(
        k,
        omega_model,
        phase,
        0.0,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )

    ratio_damped = 1.0 + damping * (ratio_linear - 1.0)

    if not return_components:
        return ratio_damped

    return ratio_damped, {
        "ratio_linear": ratio_linear,
        "damping": damping,
        "sigma": sigma,
        "omega_model": omega_model,
    }


def apply_damped_feature_to_pk(
    k,
    pk_baseline,
    z,
    omega,
    emulator,
    feature_type,
    amplitude=0.03,
    phase=0.0,
    k_pivot=0.05,
    h=0.67,
    check_domain=True,
    return_ratio=False,
):
    """
    Apply the emulator-damped feature ratio to a baseline matter power spectrum.

    This simple helper assumes the feature correction is represented as a
    multiplicative ratio relative to the supplied baseline spectrum.
    """
    k = np.asarray(k, dtype=float)
    pk_baseline = np.asarray(pk_baseline, dtype=float)

    ratio = damped_feature_ratio_from_emulator(
        k,
        z,
        omega,
        emulator=emulator,
        feature_type=feature_type,
        amplitude=amplitude,
        phase=phase,
        k_pivot=k_pivot,
        h=h,
        check_domain=check_domain,
        return_components=False,
    )

    pk_feature = pk_baseline * ratio

    if return_ratio:
        return pk_feature, ratio

    return pk_feature


def emulator_damped_power_from_spectra(
    *,
    emulator,
    redshift,
    log10omega,
    A_feat,
    phi,
    k,
    p_van_lin,
    p_van_nl,
    p_wig_lin,
    k_fit_min=0.05,
    k_fit_max=0.6,
    check_domain=True,
    h=0.67,
    propagate_gp_uncertainty=True,
):
    """
    Apply the Sigma-emulator damping correction to already-generated CAMB spectra.

    This is the cached forecast path. The vanilla spectra can be reused across
    all feature-parameter evaluations at fixed cosmology.
    """
    k = np.asarray(k, dtype=float)
    p_van_lin = np.asarray(p_van_lin, dtype=float)
    p_van_nl = np.asarray(p_van_nl, dtype=float)
    p_wig_lin = np.asarray(p_wig_lin, dtype=float)

    mask = (k >= k_fit_min) & (k <= k_fit_max)

    if not np.any(mask):
        raise ValueError(
            f"No CAMB k values inside requested range "
            f"[{k_fit_min}, {k_fit_max}]."
        )

    k = k[mask]
    p_van_lin = p_van_lin[mask]
    p_van_nl = p_van_nl[mask]
    p_wig_lin = p_wig_lin[mask]

    z_grid = np.full_like(k, float(redshift), dtype=float)
    omega_grid = np.full_like(k, float(log10omega), dtype=float)

    if propagate_gp_uncertainty:
        sigma, sigma_std = emulator.predict_sigma(
            z_grid,
            omega_grid,
            return_std=True,
            check_domain=check_domain,
        )
        damping = np.exp(-0.5 * (h * k * sigma) ** 2)
    else:    
        damping, sigma = emulator.damping(
            k,
            z_grid,
            omega_grid,
            return_sigma=True,
            check_domain=check_domain,
        )
        sigma_std = np.zeros_like(sigma, dtype=float)

    p_wig_nl = nonlinear_wiggle_power(
        p_van_lin=p_van_lin,
        p_van_nl=p_van_nl,
        p_wig_lin=p_wig_lin,
        damping=damping,
    )

    ratio_lin = p_wig_lin / p_van_lin
    ratio_nl = p_wig_nl / p_van_nl

    ratio_gp_std = np.abs(
       (ratio_nl - 1.0)
       * (h * k) ** 2
       * sigma
       * damping
       * sigma_std
    )

    power_gp_std = p_van_nl * ratio_gp_std

    return {
        "z": float(redshift),
        "log10omega": float(log10omega),
        "A_feat": float(A_feat),
        "phi": float(phi),
        "k": k,
        "p_van_lin": p_van_lin,
        "p_van_nl": p_van_nl,
        "p_wig_lin": p_wig_lin,
        "p_wig_nl": p_wig_nl,
        "ratio_lin": ratio_lin,
        "ratio_nl": ratio_nl,
        "damping": damping,
        "sigma": sigma,
        "sigma_std": sigma_std,
        "ratio_gp_std": ratio_gp_std,
        "power_gp_std": power_gp_std,
    }

def emulator_damped_camb_data_vector(
    *,
    emulator,
    vanilla_cache,
    redshifts,
    log10omega,
    A_feat,
    phi,
    feature_type="log",
    kmax=0.8,
    npoints=700,
    k_fit_min=0.05,
    k_fit_max=0.6,
    h=0.67,
    check_domain=True,
    cosmology=None,
    camb_options=None,
    observable="ratio_nl",
    propagate_gp_uncertainty=True,
):
    """
    Build a stacked forecast data vector using cached vanilla spectra and a
    single multi-redshift CAMB call for the wiggle linear spectra.
    """
    warn_if_uncalibrated_amplitude(A_feat)

    redshifts = [float(z) for z in redshifts]

    wiggle_cache = get_wiggle_linear_spectra_redshifts(
        redshifts=redshifts,
        log10omega_feat=log10omega,
        A_feat=A_feat,
        phi=phi,
        feature_type=feature_type,
        kmax=kmax,
        npoints=npoints,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    blocks = []

    for redshift in redshifts:
        if redshift not in vanilla_cache:
            raise KeyError(f"No vanilla spectra cached for redshift {redshift}.")

        vanilla = vanilla_cache[redshift]
        wiggle = wiggle_cache[redshift]

        if not np.allclose(vanilla["k"], wiggle["k"], rtol=0.0, atol=0.0):
            raise ValueError(
                f"Vanilla and wiggle CAMB k-grids differ at z={redshift}."
            )

        block = emulator_damped_power_from_spectra(
            emulator=emulator,
            redshift=redshift,
            log10omega=log10omega,
            A_feat=A_feat,
            phi=phi,
            k=vanilla["k"],
            p_van_lin=vanilla["p_van_lin"],
            p_van_nl=vanilla["p_van_nl"],
            p_wig_lin=wiggle["p_wig_lin"],
            k_fit_min=k_fit_min,
            k_fit_max=k_fit_max,
            h=h,
            check_domain=check_domain,
            propagate_gp_uncertainty=propagate_gp_uncertainty,
        )

        blocks.append(block)

    if observable not in {"ratio_nl", "p_wig_nl"}:
        raise ValueError("observable must be either 'ratio_nl' or 'p_wig_nl'.")

    vector = np.concatenate([block[observable] for block in blocks])

    if observable == "ratio_nl":
        gp_std = np.concatenate([block["ratio_gp_std"] for block in blocks])
    elif observable == "p_wig_nl":
        gp_std = np.concatenate([block["power_gp_std"] for block in blocks])
    else:
        raise ValueError("observable must be either 'ratio_nl' or 'p_wig_nl'.")

    k = np.concatenate([block["k"] for block in blocks])
    z = np.concatenate(
        [np.full_like(block["k"], block["z"], dtype=float) for block in blocks]
    )

    return {
        "vector": vector,
        "k": k,
        "z": z,
        "blocks": blocks,
        "observable": observable,
        "gp_std": gp_std,
    }


def diagonal_fractional_sigma(vector, frac_error, floor=1e-12):
    """
    Diagonal standard deviation for a simple fractional-error forecast.
    """
    vector = np.asarray(vector, dtype=float)
    sigma = frac_error * np.abs(vector)
    return np.maximum(sigma, floor)



def total_diagonal_sigma(
    vector,
    *,
    observational_sigma=None,
    frac_error=None,
    gp_sigma=None,
    model_error_floor=0.0,
    floor=1e-12,
):
    """
    Combine observational, GP/emulator, and semi-analytic model uncertainty.

    Parameters
    ----------
    vector : array-like
        Forecast data vector.
    observational_sigma : array-like, optional
        Direct observational standard deviation for each data-vector element.
    frac_error : float, optional
        Simple fractional observational error. Used only if
        observational_sigma is None.
    gp_sigma : array-like, optional
        GP-propagated standard deviation on the same observable as `vector`.
    model_error_floor : float, optional
        Additive model-error floor. For ratio observables this should be an
        absolute error on the ratio, e.g. 5e-3. For power-spectrum observables,
        pass a power-spectrum error.
    floor : float
        Minimum standard deviation.

    Returns
    -------
    sigma_total : ndarray
        Total diagonal standard deviation.
    """
    vector = np.asarray(vector, dtype=float)

    variance = np.zeros_like(vector, dtype=float)

    if observational_sigma is not None:
        obs = np.asarray(observational_sigma, dtype=float)
        if obs.shape != vector.shape:
            raise ValueError(
                f"observational_sigma shape {obs.shape} does not match "
                f"vector shape {vector.shape}."
            )
        variance += obs**2
    elif frac_error is not None:
        variance += diagonal_fractional_sigma(vector, frac_error, floor=0.0) ** 2

    if gp_sigma is not None:
        gp_sigma = np.asarray(gp_sigma, dtype=float)
        if gp_sigma.shape != vector.shape:
            raise ValueError(
                f"gp_sigma shape {gp_sigma.shape} does not match "
                f"vector shape {vector.shape}."
            )
        variance += gp_sigma**2

    if model_error_floor is not None and model_error_floor > 0.0:
        variance += float(model_error_floor) ** 2

    return np.maximum(np.sqrt(variance), floor)


def gaussian_loglike_diagonal(model, data, sigma):
    """
    Gaussian log-likelihood with diagonal covariance, excluding constants.

    Returns
    -------
    loglike : float
        -0.5 chi^2.
    chi2 : float
        Chi-square value.
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
    chi2 = np.sum(resid**2)

    return -0.5 * chi2, chi2