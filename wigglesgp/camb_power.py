import numpy as np


def primordial_pk_feature(
    k,
    *,
    As,
    ns,
    A_feat=0.0,
    log10omega_feat=1.0,
    phi=0.0,
    feature_type="none",
    k_pivot=0.05,
):
    """
    Primordial scalar power spectrum with optional oscillatory features.
    """
    k = np.asarray(k, dtype=np.float64)
    omega = 10.0 ** float(log10omega_feat)

    pk = As * (k / k_pivot) ** (ns - 1.0)

    if feature_type == "log":
        modulation = 1.0 + A_feat * np.cos(
            omega * np.log(k / k_pivot) + phi
        )
    elif feature_type == "linear":
        modulation = 1.0 + A_feat * np.cos(
            omega * (k - k_pivot) + phi
        )
    elif feature_type == "none":
        modulation = 1.0
    else:
        raise ValueError(
            f"Invalid feature_type: {feature_type}. "
            "Expected 'log', 'linear', or 'none'."
        )

    pk = pk * modulation

    if np.any(~np.isfinite(pk)):
        raise ValueError("Primordial power spectrum contains non-finite values.")

    if np.any(pk <= 0.0):
        raise ValueError(
            "Primordial power spectrum contains non-positive values. "
            "Check A_feat and other feature parameters."
        )

    return pk.item() if pk.ndim == 0 else pk


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
    feature_type="none",
    A_feat=0.0,
    log10omega_feat=1.0,
    phi=0.0,
    k_pivot=0.05,
    pk_kmin=1e-7,
    pk_kmax=100.0,
    pk_N_min=8000,
    pk_rtol=1e-12,
):
    """
    Build CAMBparams for vanilla or primordial-feature matter spectra.
    """
    import camb
    from camb import model

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

    pars.set_dark_energy(dark_energy_model=dark_energy_model)

    pars.set_matter_power(
        redshifts=[float(z) for z in redshifts],
        kmax=float(kmax),
    )

    if nonlinear:
        pars.NonLinear = model.NonLinear_both
        try:
            pars.NonLinearModel.set_params(halofit_version=halofit_version)
        except Exception:
            pass
    else:
        pars.NonLinear = model.NonLinear_none

    def primordial_pk_for_camb(k):
        return primordial_pk_feature(
            k,
            As=As,
            ns=ns,
            A_feat=A_feat,
            log10omega_feat=log10omega_feat,
            phi=phi,
            feature_type=feature_type,
            k_pivot=k_pivot,
        )

    pars.set_initial_power_function(
        primordial_pk_for_camb,
        args=(),
        kmin=pk_kmin,
        kmax=pk_kmax,
        N_min=pk_N_min,
        rtol=pk_rtol,
    )

    pars.InitPower.effective_ns_for_nonlinear = ns

    return pars


def get_matter_power_redshifts(
    *,
    redshifts,
    kmax=0.8,
    npoints=500,
    nonlinear=False,
    feature_type="none",
    A_feat=0.0,
    log10omega_feat=1.0,
    phi=0.0,
    cosmology=None,
    camb_options=None,
):
    """
    Compute CAMB matter power spectra for multiple redshifts in one CAMB call.
    """
    import camb

    cosmology = dict(cosmology or {})
    camb_options = dict(camb_options or {})

    pars = make_camb_params(
        redshifts=[float(z) for z in redshifts],
        kmax=float(kmax),
        nonlinear=nonlinear,
        feature_type=feature_type,
        A_feat=A_feat,
        log10omega_feat=log10omega_feat,
        phi=phi,
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

    if pk.ndim != 2:
        raise RuntimeError(
            f"Expected CAMB pk output with shape (n_z, n_k), got {pk.shape}."
        )

    if pk.shape[0] != len(z_out):
        raise RuntimeError(
            f"CAMB returned pk shape {pk.shape} but z_out has length {len(z_out)}."
        )

    return {
        "k": kh,
        "z": z_out,
        "pk": pk,
    }


def _match_redshift_index(z_out, redshift, atol=1e-8):
    """
    Return the row index in CAMB's output corresponding to a requested redshift.

    CAMB may reorder redshifts internally, so never assume the output order
    matches the input order.
    """
    z_out = np.asarray(z_out, dtype=float)
    redshift = float(redshift)

    matches = np.where(np.isclose(z_out, redshift, rtol=0.0, atol=atol))[0]

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely match redshift {redshift} in CAMB output "
            f"redshifts {z_out}."
        )

    return int(matches[0])


def _spectra_cache_from_camb_result(result, redshifts, value_name):
    """
    Convert a multi-redshift CAMB result into a cache keyed by requested redshift.
    """
    cache = {}

    for redshift in redshifts:
        redshift = float(redshift)
        idx = _match_redshift_index(result["z"], redshift)

        cache[redshift] = {
            "k": result["k"],
            value_name: result["pk"][idx],
        }

    return cache


def build_vanilla_spectra_cache(
    *,
    redshifts,
    kmax=0.8,
    npoints=500,
    cosmology=None,
    camb_options=None,
):
    """
    Build vanilla linear and non-linear spectra for all redshifts.

    This uses two CAMB calls total: one linear and one non-linear.
    """
    redshifts = [float(z) for z in redshifts]

    lin = get_matter_power_redshifts(
        redshifts=redshifts,
        kmax=kmax,
        npoints=npoints,
        nonlinear=False,
        feature_type="none",
        A_feat=0.0,
        log10omega_feat=1.0,
        phi=0.0,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    nl = get_matter_power_redshifts(
        redshifts=redshifts,
        kmax=kmax,
        npoints=npoints,
        nonlinear=True,
        feature_type="none",
        A_feat=0.0,
        log10omega_feat=1.0,
        phi=0.0,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    if not np.allclose(lin["k"], nl["k"], rtol=0.0, atol=0.0):
        raise ValueError("Vanilla linear and non-linear CAMB k-grids differ.")

    lin_cache = _spectra_cache_from_camb_result(
        lin,
        redshifts,
        "p_van_lin",
    )

    nl_cache = _spectra_cache_from_camb_result(
        nl,
        redshifts,
        "p_van_nl",
    )

    cache = {}

    for redshift in redshifts:
        redshift = float(redshift)

        cache[redshift] = {
            "k": lin_cache[redshift]["k"],
            "p_van_lin": lin_cache[redshift]["p_van_lin"],
            "p_van_nl": nl_cache[redshift]["p_van_nl"],
        }

    return cache


def get_wiggle_linear_spectra_redshifts(
    *,
    redshifts,
    log10omega_feat,
    A_feat=0.03,
    phi=0.0,
    feature_type="log",
    kmax=0.8,
    npoints=500,
    cosmology=None,
    camb_options=None,
):
    """
    Generate linear CAMB spectra with a primordial feature for all redshifts.

    This uses one CAMB call total for the supplied redshift list.
    """
    redshifts = [float(z) for z in redshifts]

    result = get_matter_power_redshifts(
        redshifts=redshifts,
        kmax=kmax,
        npoints=npoints,
        nonlinear=False,
        feature_type=feature_type,
        A_feat=A_feat,
        log10omega_feat=log10omega_feat,
        phi=phi,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    return _spectra_cache_from_camb_result(
        result,
        redshifts,
        "p_wig_lin",
    )


def get_vanilla_and_wiggle_spectra(
    *,
    redshift,
    log10omega_feat,
    A_feat=0.03,
    phi=0.0,
    feature_type="log",
    kmax=0.8,
    npoints=500,
    cosmology=None,
    camb_options=None,
):
    """
    Generate P_van_lin, P_van_nl, and P_wig_lin for one redshift.

    This is retained for standalone tests. Internally it uses the same
    multi-redshift machinery as the forecast path.
    """
    redshift = float(redshift)

    vanilla_cache = build_vanilla_spectra_cache(
        redshifts=[redshift],
        kmax=kmax,
        npoints=npoints,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    wiggle_cache = get_wiggle_linear_spectra_redshifts(
        redshifts=[redshift],
        log10omega_feat=log10omega_feat,
        A_feat=A_feat,
        phi=phi,
        feature_type=feature_type,
        kmax=kmax,
        npoints=npoints,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    vanilla = vanilla_cache[redshift]
    wiggle = wiggle_cache[redshift]

    if not np.allclose(vanilla["k"], wiggle["k"], rtol=0.0, atol=0.0):
        raise ValueError("Vanilla and wiggle CAMB k-grids differ.")

    return {
        "k": vanilla["k"],
        "p_van_lin": vanilla["p_van_lin"],
        "p_van_nl": vanilla["p_van_nl"],
        "p_wig_lin": wiggle["p_wig_lin"],
    }