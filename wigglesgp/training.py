import numpy as np
import csv


def build_sigma_training_set(
    z,
    omega,
    sigma,
    sigma_error=None,
    feature_type=None,
):
    """
    Build training arrays for a GP emulator of Sigma(z, omega).

    Parameters
    ----------
    z : array-like, shape (N,)
        Redshifts values.
    omega : array-like
        Feature frequency values.
    sigma : array-like
        Fitted damping scales.
    sigma_error : array-like, optional
        One-sigma uncertainties on the fitted damping scales.
    feature_type: str, optional
        Metadata label, e.g "log" or "linear".
    

    Returns
    -------
    X : ndarray, shape (n_samples, 2)
        GP inputs with columns [z, omega].
    y : ndarray, shape (n_samples,)
        GP targets Sigma(z, omega).
    yerr: ndarray or None
        GP target uncertainties, if supplied.
    metadata: dict
        Basic metadata describing the training set.
    """
    z = np.asarray(z, dtype=float)
    omega = np.asarray(omega, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if not (z.shape == omega.shape == sigma.shape):
        raise ValueError(
            "z, omega, and sigma must have matching shapes: "
            f"got {z.shape}, {omega.shape}, and {sigma.shape}."
        )
    
    mask = np.isfinite(z) & np.isfinite(omega) & np.isfinite(sigma)

    if sigma_error is not None:
        sigma_error = np.asarray(sigma_error, dtype=float)
        if sigma_error.shape != sigma.shape:
            raise ValueError(
                "sigma_error must have the same shape as sigma: "
                f"got {sigma_error.shape} and {sigma.shape}."                
            )
        mask &= np.isfinite(sigma_error) & (sigma_error > 0)

    X = np.column_stack((z[mask], omega[mask]))
    y = sigma[mask]

    if sigma_error is not None:
        yerr = sigma_error[mask]
    else:
        yerr = None

    if X.shape[0] == 0:
        raise ValueError("No finite training samples remain after masking")
        

    metadata = {
        "feature_type": feature_type,
        "n_samples": int(X.shape[0]),
        "z_min": float(np.min(X[:, 0])),
        "z_max": float(np.max(X[:, 0])),
        "omega_min": float(np.min(X[:, 1])),
        "omega_max": float(np.max(X[:, 1])),
    }

    return X, y, yerr, metadata

def load_sigma_table(path, feature_type=None):
    """
    Load a fitted Sigma table from CSV.

    Returns
    -------
    z, omega, sigma, sigma_error: ndarray
    """

    rows = []

    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if feature_type is not None and row.get("feature_type") != feature_type:
                continue
                
            rows.append(row)
        
    if len(rows) == 0:
        raise ValueError(f"No rows found in Sigma table: {path}")

    z = np.array([float(row["z"]) for row in rows], dtype=float)
    omega = np.array([float(row["omega"]) for row in rows], dtype=float)
    sigma = np.array([float(row["sigma"]) for row in rows], dtype=float)

    sigma_err_value = rows[0].get("sigma_err", "")

    if sigma_err_value.lower() not in ("", "nan","none"):
        sigma_error = np.array([float(row["sigma_err"]) for row in rows], dtype=float)
    else:
        sigma_error = None
    
    return z, omega, sigma, sigma_error