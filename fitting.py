import numpy as np
from scipy import optimize as opt
from .damping import damped_wiggle_ratio


def _safe_std(std):
    std = np.asarray(std, dtype=float)

    if std.size == 0:
        return std
    
    floor = 1e-12 * np.median(std)
    if not np.isfinite(floor) or floor <= 0.0:
        floor = 1e-12
    
    return std + floor

def damping_fit_residuals(
    fit_params, 
    y, 
    k, 
    omega, 
    std=None, 
    phase_fixed=None, 
    feature_type=None,
    amplitude=0.03, 
    k_pivot=0.05, 
    h=0.67,
):
    """
    Weighted residuals for fitting the semi-analytic damping scale.

    If 'phase_fixed' is None, both sigma and pahse are fitted. Otherwise,
    only sigma is iftted and the phase is held fixed.
    """
    if phase_fixed is None:
        sigma, phase = fit_params
    else:
        sigma = fit_params[0]
        phase = phase_fixed

    model = damped_wiggle_ratio(
        k,
        omega,
        phase,
        sigma,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )
    residual = model - np.asarray(y, dtype=float)

    if std is None:
        return residual
    
    return residual / _safe_std(std)


def weighted_chi2(
    omega,
    phase,
    sigma,
    k,
    y,
    std,
    feature_type=None,
    amplitude=0.03,
    k_pivot=0.05,
    h=0.67,       
):
    """
    Weighted chi-square for a fixed damping scale and phase.
    """
    residual = damping_fit_residuals(
        [sigma],
        y,
        k,
        omega,
        std,
        phase_fixed=phase,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )
    return float(np.sum(residual**2))



def fit_damping_scale(
    k,
    y,
    omega,
    variance,
    phase_seed=0.0,
    phase_free=False,
    feature_type=None,
    amplitude=0.03,
    k_pivot=0.05,
    h=0.67,
    sigma_seed=1.0,
    max_nfev=1000,
):
    """
    Fit the semi-analytic damping scale sigma to a simulated power-spectrum ratio.

    Parameters
    ----------
    k : array-like
        Wavenumbers.
    y : array-like
        Simulated relative matter power spectrum ratio.
    omega : float
        Feature frequency.
    variance : array-like
        Per-k variance of the simulated ratio.
    phase_seed : float, optional
        Initial phase value, or fixed phase if 'phase_free' is False.
    phase_free : bool, optional
        If True, fit both sigma and phase. If False, fit sigma only.
    feature_type : {"log", "linear"}
        Feature template.
    """
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    std = np.sqrt(np.asarray(variance, dtype=float))

    if phase_free:
        x0 = np.array([sigma_seed, phase_seed], dtype=float)
        lower_bounds = [0.0, 0.0]
        upper_bounds = [np.inf, 2*np.pi]
        phase_fixed = None
        n_parameters = 2
    else:
        x0 = np.array([sigma_seed], dtype=float)
        lower_bounds = [0.0]
        upper_bounds = [np.inf]
        phase_fixed = phase_seed
        n_parameters = 1
    
    result = opt.least_squares(
        fun=damping_fit_residuals,
        x0=x0,
        args=(y, k, omega, std),
        kwargs={
            "phase_fixed": phase_fixed,
            "feature_type": feature_type,
            "amplitude": amplitude,
            "k_pivot": k_pivot,
            "h": h,
        },
        bounds=(lower_bounds, upper_bounds),
        x_scale='jac',
        ftol=1e-8, 
        xtol=1e-8, 
        gtol=1e-8,
        max_nfev=max_nfev,
        verbose=0
    )

    if not result.success:
        raise RuntimeError(f"Damping fit failed failed: {result.message}")
    
    sigma_hat = float(result.x[0])
    phase_hat = float(result.x[1]) if phase_free else float(phase_fixed)

    chi2 = float(np.sum(result.fun**2))
    ndof = max(1, y.size - n_parameters)
    chi2_red = chi2 / ndof

    model = damped_wiggle_ratio(
        k,
        omega,
        phase_hat,
        sigma_hat,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )

    residual = model - y

    return{
        "results": result,
        "sigma": sigma_hat,
        "phase": phase_hat,
        "std": std,
        "chi2": chi2,
        "ndof": ndof,
        "chi2_red": chi2_red,
        "model": model,
        "residual": residual,
    }

def profile_chi2_over_sigma(
        sigma,
        omega,
        k, 
        y,
        std,
        phase_free,
        phase_seed,
        feature_type=None,
        amplitude=0.03,
        k_pivot=0.05,
        h=0.67,
):
    """
    Profile chi-square as a function of sigma.

    If 'phase_free=True', the phase is re-fitted at fixed sigma.
    """
    sigma = float(max(0.0, sigma))

    if not phase_free:
        return weighted_chi2(
            omega,
            float(phase_seed),
            sigma,
            k,
            y,
            std,
            feature_type=feature_type,
            amplitude=amplitude,
            k_pivot=k_pivot,
            h=h,
        )
    def phase_residuals(phase_array):
        phase = float(phase_array[0])
        return damping_fit_residuals(
            [sigma],
            y,
            k,
            omega,
            std,
            phase_fixed=phase,
            feature_type=feature_type,
            amplitude=amplitude,
            k_pivot=k_pivot,
            h=h,
        )
    result = opt.least_squares(
        fun=phase_residuals,
        x0=np.array([phase_seed], dtype=float),
        bounds=([0.0], [2.0 * np.pi]),
        x_scale='jac',
        ftol=1e-10, 
        xtol=1e-10, 
        gtol=1e-10,
        max_nfev=2000,
        verbose=0
    )

    phase_hat = float(result.x[0])

    return weighted_chi2(
        omega,
        phase_hat,
        sigma,
        k,
        y,
        std,
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h,
    )


def sigma_errors_delta_chi2(
        omega, 
        k, 
        y, 
        std, 
        sigma_hat, 
        phase_hat,
        phase_free, 
        phase_seed=None,
        feature_type=None,
        amplitude=0.03,
        k_pivot=0.05,
        h=0.67,
        tol=1e-8, 
        max_iter=80
):
    """
    Estimate one-sigma errors on Sigma using delta chi-square=1
    """    
    if phase_seed is None:
        phase_seed = phase_hat

    chi2_min = weighted_chi2(
        omega, 
        phase_hat, 
        sigma_hat, 
        k, 
        y, 
        std, 
        feature_type=feature_type,
        amplitude=amplitude,
        k_pivot=k_pivot,
        h=h
    )
    target = chi2_min + 1.0

    def shifted_chi2(sigma):
        return profile_chi2_over_sigma(
            sigma,
            omega,
            k, 
            y,
            std,
            phase_free,
            phase_seed,
            feature_type=feature_type,
            amplitude=amplitude,
            k_pivot=k_pivot,
            h=h,
        ) - target

    def bracket_and_bisect(direction):
        step = max(1e-8, 1e-3 * max(1.0, abs(sigma_hat)))
        left = sigma_hat
        f_left = shifted_chi2(left)

        right = max(0.0, sigma_hat - step) if direction < 0 else sigma_hat + step
        f_right = shifted_chi2(right)


        for _ in range(max_iter):
            if np.sign(f_left) != np.sign(f_right):
                break
            step *= 1.8
            right = max(0.0, sigma_hat - step) if direction < 0 else sigma_hat + step
            f_right = shifted_chi2(right)
            if direction < 0 and right <= 0.0 and np.sign(f_left) == np.sign(f_right):
                return np.nan
        else:
            return np.nan

        a, fa = left, f_left
        b, fb = right, f_right

        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            fm = shifted_chi2(mid)

            if abs(fm) < 1e-10 or abs(b - a) < tol * (1.0 + abs(mid)):
                return mid
            
            if np.sign(fa) * np.sign(fm) <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        return 0.5 * (a + b)

    sigma_plus  = bracket_and_bisect(+1)
    sigma_minus = bracket_and_bisect(-1)

    dpos = sigma_plus - sigma_hat  if np.isfinite(sigma_plus)  else np.nan
    dneg = sigma_hat - sigma_minus if np.isfinite(sigma_minus) else np.nan
    
    if np.all(np.isfinite([dpos, dneg])):
        dsym = 0.5 * (dpos + dneg)
    else:
        dsym = np.nan
        
    return dpos, dneg, dsym