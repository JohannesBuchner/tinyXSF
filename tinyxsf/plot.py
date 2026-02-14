"""Routines for visualisation of the fit."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.special import gammainccinv as _gammainccinv
except Exception:  # scipy may be absent in some environments
    _gammainccinv = None


def _sigma_to_quantiles(sigma):
    """
    Convert Gaussian-equivalent sigma to central-interval quantiles.

    Parameters
    ----------
    sigma : int
        Sigma-equivalent level. Supported: 1, 2, 3.

    Returns
    -------
    qlo, qhi : float
        Lower and upper quantiles for a central interval.
    """
    # Central coverage for a normal distribution:
    # 1σ: 0.6826894921
    # 2σ: 0.9544997361
    # 3σ: 0.9973002039
    cov = {1: 0.6826894921370859,
           2: 0.9544997361036416,
           3: 0.9973002039367398}
    if sigma not in cov:
        raise ValueError("sigma must be one of {1,2,3}")
    cl = cov[sigma]
    qlo = 0.5 - 0.5 * cl
    qhi = 0.5 + 0.5 * cl
    return qlo, qhi


def _poisson_yerr_from_counts(k, sigma=1):
    """
    Compute Poisson error bars from counts.

    Uses a central credible interval for the Poisson mean based on
    `gammainccinv(k+1, q)` if SciPy is available. If SciPy is not available,
    falls back to a symmetric approximation `sqrt(k + 1)`.

    Parameters
    ----------
    k : array_like
        Observed counts per bin (non-negative).
    sigma : int, optional
        Sigma-equivalent level for the interval. Supported: 1, 2, 3.

    Returns
    -------
    yerr : ndarray
        Asymmetric y-errors with shape (2, N): [lower, upper].
        For the fallback, lower==upper==sqrt(k+1).
    """
    k = np.asarray(k, dtype=float)
    if np.any(k < 0):
        raise ValueError("Counts must be non-negative.")

    if _gammainccinv is None:
        e = np.sqrt(k + 1.0)
        return np.vstack([e, e])

    qlo, qhi = _sigma_to_quantiles(sigma)
    # Interval for the Poisson mean (lambda) under flat prior:
    # Many astro codes use gammainccinv(k+1, q) with swapped q's for lower/upper.
    lo = _gammainccinv(k + 1.0, qhi)
    hi = _gammainccinv(k + 1.0, qlo)
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, 0.0)

    err_lo = k - lo
    err_hi = hi - k
    err_lo = np.maximum(err_lo, 0.0)
    err_hi = np.maximum(err_hi, 0.0)
    return np.vstack([err_lo, err_hi])


def _rebin_counts_by_snr(e_lo, e_hi, counts1d, maxbins=10, minsig=3.0):
    """
    Rebin adjacent bins until reaching a minimum S/N or a max bin count.

    Parameters
    ----------
    e_lo : array_like
        Lower edges of original bins.
    e_hi : array_like
        Upper edges of original bins.
    counts1d : array_like
        Counts per original bin (non-negative).
    maxbins : int, optional
        Maximum number of original bins to combine.
    minsig : float or None, optional
        Minimum S/N threshold using Poisson S/N ~ sqrt(N). If None, no S/N target.

    Returns
    -------
    e_lo_new, e_hi_new, counts_new : ndarray
        Rebinned edges and summed counts.
    """
    e_lo = np.asarray(e_lo)
    e_hi = np.asarray(e_hi)
    c = np.asarray(counts1d, dtype=float)

    out_lo, out_hi, out_c = [], [], []
    i, n = 0, len(c)
    while i < n:
        j = i
        csum = 0.0
        while True:
            csum += c[j]
            j += 1
            sn = csum / np.sqrt(max(csum, 1.0))  # ~sqrt(csum)
            reached_sig = (minsig is not None) and (sn >= minsig)
            reached_max = (j - i) >= maxbins
            reached_end = (j >= n)
            if reached_sig or reached_max or reached_end:
                out_lo.append(e_lo[i])
                out_hi.append(e_hi[j - 1])
                out_c.append(csum)
                break
        i = j

    return np.array(out_lo), np.array(out_hi), np.array(out_c)


def _sum_into_new_bins(values, e_lo_orig, e_hi_orig, e_lo_new, e_hi_new):
    """
    Sum per-channel values into a new binning defined by (e_lo_new, e_hi_new).

    Parameters
    ----------
    values : ndarray
        Either shape (Nchan,) or (Npost, Nchan).
    e_lo_orig : ndarray
        Original lower bin edges (masked channel grid).
    e_hi_orig : ndarray
        Original upper bin edges (masked channel grid).
    e_lo_new : ndarray
        New lower bin edges (must align to original bin edges used to define sums).
    e_hi_new : ndarray
        New lower bin edges (must align to original bin edges used to define sums).

    Returns
    -------
    out : ndarray
        Summed values in new bins. Shape is (Nnew,) or (Npost, Nnew).
    """
    v = np.asarray(values, dtype=float)
    two_d = (v.ndim == 2)
    if not two_d:
        v = v[None, :]

    out = np.zeros((v.shape[0], len(e_lo_new)), dtype=float)
    for k, (lo, hi) in enumerate(zip(e_lo_new, e_hi_new)):
        i0 = np.searchsorted(e_lo_orig, lo, side="left")
        i1 = np.searchsorted(e_hi_orig, hi, side="right")
        out[:, k] = v[:, i0:i1].sum(axis=1)

    return out if two_d else out[0]


def _avg_into_new_bins(vec, e_lo_orig, e_hi_orig, e_lo_new, e_hi_new):
    """
    Width-weighted average of a per-channel vector into a new binning.

    Parameters
    ----------
    vec : ndarray, shape (Nchan,)
        Per-channel vector to rebin (e.g. throughput-like divisor).
    e_lo_orig : ndarray
        Original lower bin edges.
    e_hi_orig : ndarray
        Original upper bin edges.
    e_lo_new : ndarray
        New lower bin edges.
    e_hi_new : ndarray
        New upper bin edges.

    Returns
    -------
    out : ndarray, shape (Nnew,)
        Rebinned vector values (weighted average by original bin widths).
    """
    vec = np.asarray(vec, dtype=float)
    out = np.zeros(len(e_lo_new), dtype=float)
    for k, (lo, hi) in enumerate(zip(e_lo_new, e_hi_new)):
        i0 = np.searchsorted(e_lo_orig, lo, side="left")
        i1 = np.searchsorted(e_hi_orig, hi, side="right")
        w = (e_hi_orig[i0:i1] - e_lo_orig[i0:i1])
        out[k] = np.average(vec[i0:i1], weights=w) if np.sum(w) > 0 else np.nan
    return out


def plot_fit(
    data,
    pred_counts_arrays,
    labels,
    colors,
    *,
    subplot="normalised_residuals",
    cumulative=False,
    counts=False,
    rebinsig=3.0,
    rebinbnum=10,
    divide_vector=None,
    sigma=1,
    show_bands=True,
    plot_bkg_data=True,
    exposure=None,
    ax=None,
    ax_sub=None,
):
    """
    Plot spectral data with posterior prediction bands and a diagnostic panel.

    Parameters
    ----------
    data : dict
        Spectrum container. Required keys:
        'chan_mask', 'chan_e_min', 'chan_e_max', 'src_region_counts'.
        If `plot_bkg_data=True`, also requires 'bkg_region_counts'.
        If `counts=False` and `exposure` is None, requires 'src_exposure' [s].
    pred_counts_arrays : sequence of ndarray
        Model predictions for *source-region* counts per channel, aligned to
        `data['chan_mask']`. Each array has shape (npost, nchan_masked).
        The first entry is used as the reference model for the lower panel.
    labels : sequence
        Labels colors for each model array.
    colors : sequence
        matplotlib colors for each model array.
    subplot : {'residuals', 'normalised_residuals', 'ratio'}, optional
        Lower panel content:
        - 'residuals': data - model (in plotted y-units)
        - 'normalised_residuals': (data - model) / sigma_data
        - 'ratio': data / model
        Default is 'normalised_residuals'.
    cumulative : bool, optional
        If True, plot cumulative sums over energy (low->high). Default False.
    counts : bool, optional
        If True, plot integer counts per bin. If False, plot Counts/s/keV.
        Default False.
    rebinsig : float or None, optional
        Plotting-only rebinning target S/N using Poisson S/N ~ sqrt(N) on
        source-region counts. If None, do not rebin. Default 3.0.
    rebinbnum : int, optional
        Maximum number of bins to combine during rebinning. Default 10.
    divide_vector : ndarray or None, optional
        Optional per-channel vector (length nchan_masked) that is divided out
        from both data and model after rebinning. Intended for throughput-like
        factors (e.g. a folded Galactic absorption shape).
    sigma : {1, 2, 3}, optional
        Sigma-equivalent level for:
        - data error bars (Poisson interval if SciPy available, else sqrt(k+1))
        - inner posterior prediction band.
        Default 1.
    show_bands : bool, optional
        If True, draw posterior prediction bands: inner = `sigma`,
        outer = 3 sigma. Default True.
    plot_bkg_data : bool, optional
        If True, plot background-region data (rebinned to the same binning)
        with error bars, in the same y-units as the main plot. Default True.
        Note: background-region data are not scaled to source-region area here.
    exposure : float or None, optional
        Exposure time in seconds (required if `counts=False`). If None, uses
        data['src_exposure'] if present.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw into. If None, a new 2-panel figure is created.
    ax_sub : matplotlib.axes.Axes or None, optional
        Axis to draw into. If None, a new 2-panel figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : tuple of matplotlib.axes.Axes
        (ax_main, ax_sub) axes.
    """
    e_lo_c = data["chan_e_min"]
    e_hi_c = data["chan_e_max"]
    counts_src_c = data["src_region_counts"]

    if exposure is None and (not counts):
        if "src_expo" not in data:
            raise KeyError("Need `exposure` or data['src_expo'] when counts=False.")
        exposure = float(data["src_expo"])

    if divide_vector is not None:
        divide_vector = np.asarray(divide_vector, dtype=float)
        if divide_vector.shape != counts_src_c.shape:
            raise ValueError("divide_vector must have length equal to number of masked channels.")

    # plotting binning
    if rebinsig is None:
        e_lo_p, e_hi_p, counts_src_p = e_lo_c, e_hi_c, counts_src_c
    else:
        e_lo_p, e_hi_p, counts_src_p = _rebin_counts_by_snr(
            e_lo_c, e_hi_c, counts_src_c, maxbins=rebinbnum, minsig=rebinsig
        )

    e_mid_p = 0.5 * (e_lo_p + e_hi_p)
    de_p = (e_hi_p - e_lo_p)
    xerr_p = np.vstack([e_mid_p - e_lo_p, e_hi_p - e_mid_p])

    # divisor rebinned
    div_p = None
    if divide_vector is not None:
        div_p = _avg_into_new_bins(divide_vector, e_lo_c, e_hi_c, e_lo_p, e_hi_p)
        div_p = np.maximum(div_p, 1e-30)

    # data errors in counts (asymmetric if SciPy available)
    yerr_counts = _poisson_yerr_from_counts(counts_src_p, sigma=sigma)

    # convert to y-units
    if counts:
        y_data = counts_src_p.copy()
        yerr_data = yerr_counts.copy()
    else:
        y_data = counts_src_p / (exposure * de_p)
        yerr_data = yerr_counts / (exposure * de_p)[None, :]

    if div_p is not None:
        y_data = y_data / div_p
        yerr_data = yerr_data / div_p[None, :]

    # cumulative if requested
    if cumulative:
        y_data = np.cumsum(y_data)
        yerr_data = None
        xerr_plot = None
    else:
        xerr_plot = xerr_p

    # background data (optional)
    y_bkg = yerr_bkg = None
    if plot_bkg_data:
        if "bkg_region_counts" not in data:
            raise KeyError("plot_bkg_data=True requires data['bkg_region_counts'].")
        counts_bkg_c = data["bkg_region_counts"]
        counts_bkg_p = _sum_into_new_bins(counts_bkg_c, e_lo_c, e_hi_c, e_lo_p, e_hi_p)

        bkg_yerr_counts = _poisson_yerr_from_counts(counts_bkg_p, sigma=sigma)

        if counts:
            y_bkg = counts_bkg_p * data['src_to_bkg_ratio']
            yerr_bkg = bkg_yerr_counts * data['src_to_bkg_ratio']
        else:
            y_bkg = counts_bkg_p / (exposure * de_p) * data['src_to_bkg_ratio']
            yerr_bkg = bkg_yerr_counts / (exposure * de_p)[None, :] * data['src_to_bkg_ratio']

        if div_p is not None:
            y_bkg = y_bkg / div_p
            yerr_bkg = yerr_bkg / div_p[None, :]

        if cumulative:
            y_bkg = np.cumsum(y_bkg)
            yerr_bkg = np.vstack([np.sqrt(np.cumsum(yerr_bkg[0] ** 2)),
                                  np.sqrt(np.cumsum(yerr_bkg[1] ** 2))])

    # axes
    if ax is None or ax_sub is None:
        fig, (ax, ax_sub) = plt.subplots(
            2, 1, figsize=(5, 6), sharex=True,
            gridspec_kw=dict(height_ratios=[4, 1], hspace=0.02)
        )
    else:
        fig = ax.figure

    # plot data
    ax.errorbar(
        e_mid_p, y_data, yerr=yerr_data, xerr=xerr_plot,
        fmt="o", ms=4, mfc="none", mec="k", ecolor="k", elinewidth=1, capsize=0,
        zorder=10, label="data"
    )

    if y_bkg is not None:
        ax.errorbar(
            e_mid_p, y_bkg, yerr=yerr_bkg, xerr=xerr_plot,
            fmt="s", ms=3.5, mfc="none", mec="0.7", ecolor="0.7", elinewidth=1, capsize=0,
            zorder=9, label="bkg data"
        )

    # models + bands
    q_in_lo, q_in_hi = _sigma_to_quantiles(sigma)
    q_out_lo, q_out_hi = _sigma_to_quantiles(3)

    model_medians = []
    for pred_counts_srcreg, label, color in zip(pred_counts_arrays, labels, colors):
        pred_counts_reb = _sum_into_new_bins(pred_counts_srcreg, e_lo_c, e_hi_c, e_lo_p, e_hi_p)  # (npost, nbin)

        if counts:
            y_model = pred_counts_reb
        else:
            y_model = pred_counts_reb / (exposure * de_p[None, :])

        if div_p is not None:
            y_model = y_model / div_p[None, :]

        if cumulative:
            y_model = np.cumsum(y_model, axis=1)

        med = np.nanmedian(y_model, axis=0)
        model_medians.append(med)

        if show_bands:
            lo_in = np.quantile(y_model, q_in_lo, axis=0)
            hi_in = np.quantile(y_model, q_in_hi, axis=0)
            lo_out = np.quantile(y_model, q_out_lo, axis=0)
            hi_out = np.quantile(y_model, q_out_hi, axis=0)

            ax.fill_between(e_mid_p, lo_out, hi_out, color=color, alpha=0.08, lw=0)
            ax.fill_between(e_mid_p, lo_in, hi_in, color=color, alpha=0.25, lw=0)

        ax.plot(e_mid_p, med, color=color, lw=1.5, label=label)

    # lower panel vs first model
    if len(model_medians) == 0:
        raise ValueError("pred_counts_arrays must contain at least one model array.")
    mref = np.asarray(model_medians[0], dtype=float)

    if yerr_data is not None:
        sigma_y = 0.5 * (yerr_data[0] + yerr_data[1])
        sigma_y = np.where(sigma_y > 0, sigma_y, np.nan)
    else:
        sigma_y = None

    if subplot == "residuals":
        y_sub = y_data - mref
        y_sub_err = None
        ax_sub.axhline(0.0, color="0.4", ls="--", lw=1.2)
        ax_sub.set_ylabel("Data-Model")
    elif subplot == "normalised_residuals":
        if sigma_y is None:
            raise ValueError("use cumulative counts with plot_fit(subplot='residuals')")
        y_sub = (y_data - mref) / sigma_y
        y_sub_err = None
        ax_sub.axhline(0.0, color="0.4", ls="--", lw=1.2)
        ax_sub.set_ylabel(r"Residuals/$\sigma$")
        ax_sub.set_ylim(-3, 3)
        ax_sub.set_yticks([-2, 0, 2])
    elif subplot == "ratio":
        y_sub = y_data / np.where(mref != 0, mref, np.nan)
        y_sub_err = yerr_data / np.where(mref != 0, mref, np.nan)[None, :]
        ax_sub.axhline(1.0, color="0.4", ls="--", lw=1.2)
        ax_sub.set_ylabel("Data/Model")
        ax_sub.set_ylim(0, 5)
        ax_sub.set_yticks([0, 1, 3])
    else:
        raise ValueError("subplot must be one of: 'residuals', 'normalised_residuals', 'ratio'.")

    ax_sub.errorbar(
        e_mid_p, y_sub, yerr=y_sub_err, xerr=xerr_plot,
        fmt="o", ms=4, mfc="none", mec="k", ecolor="k", elinewidth=1, capsize=0
    )

    if subplot == "residuals":
        ylo, yhi = ax_sub.get_ylim()
        yhi = max(abs(ylo), abs(yhi))
        ax_sub.set_ylim(-yhi, yhi)

    # cosmetics (leave y-scale to user)
    ax.set_xscale("log")
    ax.set_xlim(e_lo_p.min(), e_hi_p.max())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax_sub.set_xscale("log")
    ax_sub.tick_params(which="both", direction="in", top=True, right=True)
    ax_sub.set_xlabel("Energy [keV]")

    ax.set_ylabel("Counts" if counts else "Counts / s / keV")

    return fig, (ax, ax_sub)
