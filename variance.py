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