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

    Parameters
    ----------
    k : float or array-like
        Wavenumber in Mpc^-1.
    As : float
        Scalar amplitude at k_pivot.
    ns : float
        Scalar spectral index.
    A_feat : float
        Feature amplitude.
    log10omega_feat : float
        Base-10 logarithm of the oscillation frequency. This is the same
        convention used by the Sigma emulator training table.
    phi : float
        Phase in radians.
    feature_type : {"log", "linear", "none"}
        Feature template.
    k_pivot : float
        Pivot scale in Mpc^-1.

    Returns
    -------
    pk : float or ndarray
        Primordial scalar power spectrum.
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
        raise ValueError(f"Invalid feature_type: {feature_type}. Expected 'log', 'linear', or 'none'.")
    
    pk = pk * modulation

    if np.any(~np.isfinite(pk)):
        raise ValueError("Primordial power spectrum contains non-finite values.")
    if np.any(pk <= 0.0):
        raise ValueError("Primordial power spectrum contains non-positive values. Check A_feat and other feature parameters.")
    
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
        redshifts=list(redshifts),
        kmax=kmax,
    )

    if nonlinear:
        pars.NonLinear = model.NonLinear_both
        try:
            pars.NonLinearModel.set_params(halofit_version=halofit_version)
        except Exception:
            # Older/newer CAMB version differ here slightly
            # The default non-linera model is still useable for the standalone test.
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

def get_matter_power(
        *,
        redshift,
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
    Computed a CAMB matter power spectrum for one redshift.

    Returns
    -------
    k : ndarray
        Wavenumbers in h/Mpc, matching CAMB's get_matter_power_spectrum output.
    pk : ndarray
        Matter power spectrum in CAMB's corresponding units.
    """
    import camb

    cosmology = dict(cosmology or {})
    camb_options = dict(camb_options or {})

    # CAMB internally may need kmax larger than the exact plotted/fitted range.
    pars = make_camb_params(
        redshifts=[float(redshift)],
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

    # For one requested redshift, pk has shape (1, n_k).
    pk = np.asarray(pk, dtype=float)
    if pk.ndim != 2 or pk.shape[0] != 1:
        raise ValueError(f"Expected one-redshift CAMB output with shape (1, n_k), got shape {pk.shape}.")

    return np.asarray(kh, dtype=float), pk[0]


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
    Generate the three spectra needed by the emulator damping correction:
        P_van_lin(k, z)
        P_van_nl(k, z)
        P_wig_lin(k, z)

    The feature spectrum is intentionally linear. The non-linear information enters trhough P_van_nl and the emulator damping envelope.        
    """

    k_van_lin, p_van_lin = get_matter_power(
        redshift=redshift,
        kmax=kmax,
        npoints=npoints,
        nonlinear=False,
        feature_type="none",
        A_feat=0.0,
        log10omega_feat=log10omega_feat,
        phi=phi,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    k_van_nl, p_van_nl = get_matter_power(
        redshift=redshift,
        kmax=kmax,
        npoints=npoints,
        nonlinear=True,
        feature_type="none",
        A_feat=0.0,
        log10omega_feat=log10omega_feat,
        phi=phi,
        cosmology=cosmology,
        camb_options=camb_options,
    )

    k_wig_lin, p_wig_lin = get_matter_power(
        redshift=redshift,
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

    if not np.allclose(k_van_lin, k_van_nl, rtol=0.0, atol=0.0):
        raise ValueError("Vanilla linear and non-linear CAMB k-grids differ.")

    if not np.allclose(k_van_lin, k_wig_lin, rtol=0.0, atol=0.0):
        raise ValueError("Vanilla linear and wiggle CAMB k-grids differ.")
    
    return {
        "k": k_van_lin,
        "p_van_lin": p_van_lin,
        "p_van_nl": p_van_nl,
        "p_wig_lin": p_wig_lin,
    }