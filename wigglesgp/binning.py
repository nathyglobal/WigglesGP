from __future__ import annotations

import numpy as np


def make_uniform_k_bin_edges(k_min=0.05, k_max=0.6, delta_k=0.004, *, drop_partial_final_bin=True):
    k_min = float(k_min)
    k_max = float(k_max)
    delta_k = float(delta_k)

    if delta_k <= 0:
        raise ValueError("delta_k must be positive.")
    if k_max <= k_min:
        raise ValueError("k_max must be larger than k_min.")

    if drop_partial_final_bin:
        n_bins = int(np.floor((k_max - k_min) / delta_k))
        if n_bins < 1:
            raise ValueError(f"No full bins fit inside [{k_min}, {k_max}] with delta_k={delta_k}.")
        edges = k_min + delta_k * np.arange(n_bins + 1)
    else:
        edges = np.arange(k_min, k_max + 0.5 * delta_k, delta_k)
        if edges[-1] < k_max:
            edges = np.append(edges, k_max)
        edges[-1] = min(edges[-1], k_max)

    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges, centres


def top_hat_bin_average_1d(k, y, bin_edges):
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if k.ndim != 1 or y.ndim != 1:
        raise ValueError("k and y must be one-dimensional.")
    if k.shape != y.shape:
        raise ValueError(f"k and y shapes differ: {k.shape} vs {y.shape}.")
    if np.any(np.diff(k) <= 0):
        raise ValueError("k must be strictly increasing.")
    if bin_edges[0] < k[0] or bin_edges[-1] > k[-1]:
        raise ValueError(
            f"Bin range [{bin_edges[0]}, {bin_edges[-1]}] extends outside "
            f"k range [{k[0]}, {k[-1]}]."
        )

    y_binned = np.empty(len(bin_edges) - 1, dtype=float)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (k > lo) & (k < hi)
        k_bin = np.concatenate(([lo], k[mask], [hi]))
        y_bin = np.concatenate(([np.interp(lo, k, y)], y[mask], [np.interp(hi, k, y)]))
        y_binned[i] = np.trapezoid(y_bin, k_bin) / (hi - lo)
    return y_binned


def top_hat_bin_average_prediction(
    prediction,
    *,
    k_min=0.05,
    k_max=0.6,
    delta_k=0.004,
    k_max_by_z=None,
    drop_partial_final_bin=True,
):
    """
    Top-hat average each redshift block into k-bandpowers.

    The binned ratio is defined as

        R_i = <P_pred>_i / <P_ref>_i,

    rather than as <R(k)>_i. This makes the ratio and power observables
    equivalent representations of the same bandpower likelihood.
    """
    observable = prediction["observable"]

    if observable not in {"ratio", "power"}:
        raise ValueError("prediction observable must be either 'ratio' or 'power'.")

    new_blocks = []

    for block in prediction["blocks"]:
        z = float(block["z"])

        if k_max_by_z is None:
            k_block_max = float(k_max)
        else:
            k_block_max = float(k_max_by_z.get(z, k_max))

        edges, centres = make_uniform_k_bin_edges(
            k_min=k_min,
            k_max=k_block_max,
            delta_k=delta_k,
            drop_partial_final_bin=drop_partial_final_bin,
        )

        k_old = np.asarray(block["k"], dtype=float)

        if edges[0] < k_old[0] or edges[-1] > k_old[-1]:
            raise ValueError(
                f"Cannot bin z={z}: requested [{edges[0]}, {edges[-1]}], "
                f"available [{k_old[0]}, {k_old[-1]}]."
            )

        new_block = dict(block)
        new_block["k"] = centres
        new_block["delta_k"] = np.diff(edges)

        # Bin-average every ordinary k-grid quantity.
        # We skip ratio/power-derived quantities that are recomputed below
        # to preserve exact power-ratio consistency.
        skip_keys = {
            "k",
            "ratio",
            "delta_pred",
            "power_gp_std",
        }

        for key, value in block.items():
            if key in skip_keys:
                continue

            if isinstance(value, np.ndarray) and value.shape == k_old.shape:
                new_block[key] = top_hat_bin_average_1d(k_old, value, edges)

        # The prediction block must contain binned p_reference and power.
        if "p_reference" not in new_block:
            raise KeyError("Binned prediction block is missing 'p_reference'.")

        if "power" not in new_block:
            raise KeyError("Binned prediction block is missing 'power'.")

        # Define the bandpower-level ratio from binned powers.
        new_block["ratio"] = new_block["power"] / new_block["p_reference"]
        new_block["delta_pred"] = new_block["ratio"] - 1.0

        # Bin GP uncertainty in ratio space as an RMS quantity.
        if "ratio_gp_std" in block:
            ratio_var = top_hat_bin_average_1d(
                k_old,
                np.asarray(block["ratio_gp_std"], dtype=float) ** 2,
                edges,
            )
            new_block["ratio_gp_std"] = np.sqrt(np.maximum(ratio_var, 0.0))
        else:
            new_block["ratio_gp_std"] = np.zeros_like(new_block["k"], dtype=float)

        # Convert the binned ratio uncertainty into power units.
        new_block["power_gp_std"] = (
            np.abs(new_block["p_reference"]) * new_block["ratio_gp_std"]
        )

        # Sigma_std is diagnostic only; bin as RMS if present.
        if "sigma_std" in block:
            sigma_var = top_hat_bin_average_1d(
                k_old,
                np.asarray(block["sigma_std"], dtype=float) ** 2,
                edges,
            )
            new_block["sigma_std"] = np.sqrt(np.maximum(sigma_var, 0.0))

        new_blocks.append(new_block)

    vector = np.concatenate([block[observable] for block in new_blocks])

    if observable == "ratio":
        gp_std = np.concatenate([block["ratio_gp_std"] for block in new_blocks])
    elif observable == "power":
        gp_std = np.concatenate([block["power_gp_std"] for block in new_blocks])
    else:
        raise RuntimeError("Unexpected observable.")

    k = np.concatenate([block["k"] for block in new_blocks])
    z = np.concatenate(
        [
            np.full_like(block["k"], block["z"], dtype=float)
            for block in new_blocks
        ]
    )

    return {
        **prediction,
        "vector": vector,
        "k": k,
        "z": z,
        "blocks": new_blocks,
        "gp_std": gp_std,
        "is_binned": True,
        "delta_k_bin": float(delta_k),
    }