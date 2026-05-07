import numpy as np

def corrected_per_k_variance(ratios):
    """
    Estimate the corrected variance of paired-simulation power-spectrum ratios.

    Parameters
    ----------
    ratios : array-like, shape (N_sim, N_k)
        Rows are independent simulation realisations. Columns are k-bins.

    Returns
    -------
    var_corr : ndarray, shape (N_k,)
        Corrected variance estimate for each k-bin.
    """
    ratios = np.asarray(ratios, dtype=float)

    if ratios.ndim != 2:
        raise ValueError(f"ratios must be a 2D array, got shape {ratios.shape}.")
    

    N_sim, _ = ratios.shape

    if N_sim <= 3:
        raise ValueError(f"Need N_sim > 3 for a positive correction factor; got N={N_sim}.")

    mean = ratios.mean(axis=0)
    var_hat = ((ratios - mean)**2).sum(axis=0) / (N_sim - 1)

    alpha = (N_sim - 3) / (N_sim - 2)
    return var_hat / alpha


def smooth_log_variance(k, variance, degree=3):
    k = np.asarray(k, dtype=float)
    variance = np.asarray(variance, dtype=float)

    mask = np.isfinite(k) & np.isfinite(variance) & (k > 0) & (variance > 0)

    if np.count_nonzero(mask) < degree:
        raise ValueError("Not enough positive finite variance points to smooth.")
    
    coeff = np.polyfit(np.log(k[mask]), np.log(variance[mask]), degree)

    smoothed = np.empty_like(variance)
    smoothed[:] = np.nan
    smoothed[mask] = np.exp(np.polyval(coeff, np.log(k[mask])))

    return smoothed