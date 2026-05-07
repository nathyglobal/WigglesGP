import numpy as np
import math



def _slice_to_k_range(k, values, k_min=None, k_max=None):
    k = np.asarray(k, dtype=float)
    values = np.asarray(values, dtype=float)

    if k_min is None and k_max is None:
        return k, values
    
    if k_min is None:
        start = 0
    else:
        start = np.abs(k - k_min).argmin()
    
    if k_max is None:
        end = len(k)
    else:
        end = np.abs(k - k_max).argmin()
    
    return values[start:end], k[start:end]

def read_primordial_power(name, delimiter=None, box_size=1024):
    """
    Read a primordial power spectrum file.

    The first column is converted using 2*pi/box_size, matching the original simulation convention.
    """
    data = np.loadtxt(name, delimiter=delimiter)
    k = data[:, 0] * (2 * np.pi / box_size)
    power = data[:, 1]
    return power, k


def read_initial_power(name, delimiter=None, k_min=0.05, k_max=0.6):
    """
    Read an initial matter power spectrum and restrict it to a k-range.
    """
    data = np.loadtxt(name, delimiter=delimiter)
    k = data[:, 0]
    power = data[:, 1]
    return _slice_to_k_range(k, power, k_min, k_max)

def read_power_ratio_snapshot(
        wiggle_path,
        vanilla_path,
        box_size=1024.0,
        k_min=0.05,
        k_max=0.6,
):
    """
    Read a pairsed wiggle/vanilla snapshot and return the power-spectrum ratio.

    Returns
    -------
    k : ndarray
        wavenumbers restricted to the requested k-range.
    ratio: ndarray
        P_wiggle(k) / P_vanilla(k).
    """
    wiggle = np.loadtxt(wiggle_path)
    vanilla= np.loaddtxt(vanilla_path)

    if wiggle.shape[0] != vanilla.shape[0]:
        raise ValueError(
            "Wiggle and vanilla files must have the same number of rows: "
            f"got {wiggle.shape[0]} and {vanilla.shape[0]}."
            )  
    k = wiggle[:, 0] * (2 * np.pi / box_size)
    ratio = wiggle[:, 1] / vanilla[:, 1]

    ratio, k = _slice_to_k_range(k, ratio, k_min, k_max)
    return ratio, k