from __future__ import annotations

import numpy as np



def make_camb_params(
    *,
    H0=67.7276,
    ombh2=0.0222221,
    omch2=0.11847,
    tau=0.0550674,
    As=2.09016e-9,
    ns=0.962633,
    redshifts=(0.0,),
    kmax=2.0,
    nonlinear=False,
    halofit_version="mead",
    num_massive_neutrinos=1,
    nnu=3.044,
    dark_energy_model="fluid",
    bbn_predictor="PArthENoPE_880.2_standard.dat",
):
    """Build CAMBparams for featureless matter spectra."""
    import camb
    from camb import model

    redshifts = [float(z) for z in redshifts]
    pars = camb.CAMBparams()

    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        tau=tau,
        num_massive_neutrinos=num_massive_neutrinos,
        nnu=nnu,
        bbn_predictor=bbn_predictor,
    )
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_dark_energy(dark_energy_model=dark_energy_model)
    pars.set_matter_power(redshifts=redshifts, kmax=float(kmax))

    if nonlinear:
        pars.NonLinear = model.NonLinear_both
        try:
            pars.NonLinearModel.set_params(halofit_version=halofit_version)
        except Exception:
            pass
    else:
        pars.NonLinear = model.NonLinear_none

    return pars


def get_matter_power_redshifts(
    *,
    redshifts,
    kmax=0.8,
    npoints=500,
    nonlinear=False,
    cosmology=None,
    camb_options=None,
):
    """
    Compute featureless CAMB matter power spectra for multiple redshifts.

    Returns a dictionary with a common k-grid in h/Mpc and pk with shape
    (n_z, n_k).
    """
    import camb

    redshifts = [float(z) for z in redshifts]
    redshifts_for_camb = sorted(redshifts, reverse=True)
    cosmology = dict(cosmology or {})
    camb_options = dict(camb_options or {})

    pars = make_camb_params(
        redshifts=redshifts_for_camb,
        kmax=float(kmax),
        nonlinear=nonlinear,
        **cosmology,
        **camb_options,
    )
    results = camb.get_results(pars)

    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=1e-4,
        maxkh=float(kmax),
        npoints=int(npoints),
    )

    kh = np.asarray(kh, dtype=float)
    z_out = np.asarray(z_out, dtype=float)
    pk = np.asarray(pk, dtype=float)

    if pk.ndim != 2 or pk.shape[0] != len(z_out):
        raise RuntimeError(
            f"Unexpected CAMB output: pk shape={pk.shape}, z_out length={len(z_out)}."
        )

    return {"k": kh, "z": z_out, "pk": pk}


def _match_redshift_index(z_out, redshift, atol=1e-8):
    z_out = np.asarray(z_out, dtype=float)
    matches = np.where(np.isclose(z_out, float(redshift), rtol=0.0, atol=atol))[0]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely match z={redshift} in CAMB redshifts {z_out}.")
    return int(matches[0])


def build_baseline_spectra_cache(
    *,
    redshifts,
    kmax=0.8,
    npoints=500,
    cosmology=None,
    camb_options=None,
):
    """
    Build the featureless linear baseline spectra for all requested redshifts.

    The damped feature model is applied to this baseline in predicting.py.
    """
    redshifts = [float(z) for z in redshifts]

    lin = get_matter_power_redshifts(
        redshifts=redshifts,
        kmax=kmax,
        npoints=npoints,
        nonlinear=False,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    cache = {}
    for redshift in redshifts:
        idx = _match_redshift_index(lin["z"], redshift)
        cache[float(redshift)] = {
            "k": lin["k"],
            "p_reference": lin["pk"][idx],
            # "p_van_lin": lin["pk"][idx],  # compatibility alias
        }
    return cache