import numpy as np

def linear_wiggle_residual(p_van_lin, p_wig_lin):
    """
    Return the fractional linear wiggle residual.

    delta_lin = (P_wig_lin - P_van_lin) / P_van_lin
    """

    p_van_lin = np.asarray(p_van_lin, dtype=float)
    p_wig_lin = np.asarray(p_wig_lin, dtype=float)

    if p_van_lin.shape != p_wig_lin.shape:
        raise ValueError(
            "p_van_lin and p_wig_lin must have the same shape: "
            f"got {p_van_lin.shape} and {p_wig_lin.shape}."
        )

    return (p_wig_lin - p_van_lin) / p_van_lin


def nonlinear_wiggle_power(
    p_van_lin,
    p_van_nl,
    p_wig_lin,
    damping,
):
    """
    Construct the damped non-linear wiggle matter power spectrum.

    P_model = P_van_nl * (1 + D * (P_wig_lin - P_van_lin) / P_van_lin)

    where D is the calibrated damping envelope.
    """
    p_van_lin = np.asarray(p_van_lin, dtype=float)
    p_van_nl = np.asarray(p_van_nl, dtype=float)
    p_wig_lin = np.asarray(p_wig_lin, dtype=float)
    damping = np.asarray(damping, dtype=float)

    if p_van_lin.shape != p_wig_lin.shape:
        raise ValueError(
            "p_van_lin and p_wig_lin must have the same shape: "
            f"got {p_van_lin.shape} and {p_wig_lin.shape}."
        )

    delta_lin = linear_wiggle_residual(p_van_lin, p_wig_lin)
    return p_van_nl * (1.0 + damping * delta_lin)