import numpy as np



def damped_log_wiggle_ratio(
    k,
    omega,
    phase,
    sigma,
    amplitude=0.03, 
    k_pivot=0.05, 
    h=0.67,
        
):
    """
    Semi-analytic damped logarithmic feature ratio.

    Parameters
    ----------
    k : array-like
        Wavenumbers in h/Mpc simulation units before multiplication by h
    omega : float
        Logarithmic feature frequency.
    phase : float
        Feature phase in radians.
    sigma : float
        Non-linear damping scale.
    amplitude : float, optional
        Feature amplitude (default: 0.03).
    k_pivot : float, optional
        Pivot scale (default: 0.05 h/Mpc).
    h : float, optional
        Reduced Hubble parameter used to convert k consistently with the simulation convention (default: 0.67).

    Returns
    -------
    ratio : ndarray
        Damped relative matter power spectrum ratio, 1 + delta(k).
    """
    k = np.asarray(k, dtype=float)
    damping = np.exp(-0.5 * (h * k * sigma) **2)
    wiggle = np.cos(omega * np.log(h * k  / k_pivot) + phase)
    return 1.0 + amplitude * wiggle * damping


def damped_linear_wiggle_ratio( 
    k,
    omega,
    phase,
    sigma,
    amplitude=0.03, 
    k_pivot=0.05, 
    h=0.67,
):
    """
    Semi-analytic damped linear frequency feature ratio.

    Parameters are the same as 'damped_log_wiggle_ratio', 
    expect that the oscillatory phase is linear in k rather than log(k)
    """
    k = np.asarray(k, dtype=float)
    damping = np.exp(-0.5 * (h * k * sigma) **2)
    wiggle = np.cos(omega * h * k  / k_pivot + phase)
    return 1.0 + amplitude * wiggle * damping


def damped_wiggle_ratio(
    k,
    omega,
    phase,
    sigma,
    feature_type=None,  # "log" or "linear"; if None, raises ValueError
    amplitude=0.03, 
    k_pivot=0.05, 
    h=0.67,
):
    """
    Dispatch to the logarithmic or linear damped feature model.
    """

    if feature_type == "log":
        return damped_log_wiggle_ratio(k, omega, phase, sigma, amplitude, k_pivot, h)
    elif feature_type == "linear":
        return damped_linear_wiggle_ratio(k, omega, phase, sigma, amplitude, k_pivot, h)
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}. Must be 'log' or 'linear'.")