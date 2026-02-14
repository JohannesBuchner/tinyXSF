"""Functions for flux and luminosity computations."""

import numpy as np
from astropy import units as u


def frac_overlap_interval(edges, lo, hi):
    """Compute overlap of bins.

    Parameters
    ----------
    edges: array
        edges of bins.
    lo: float
        lower limit
    hi: float
        upper limit

    Returns
    -------
    weights: array
        for each bin, what fraction of the bin lies between lo and hi.
    """
    weights = np.zeros(len(edges) - 1)
    for i, (edge_lo, edge_hi) in enumerate(zip(edges[:-1], edges[1:])):
        if edge_lo > lo and edge_hi < hi:
            weight = 1
        elif edge_hi < lo:
            weight = 0
        elif edge_lo > hi:
            weight = 0
        elif hi > edge_hi and lo > edge_lo:
            weight = (edge_hi - lo) / (edge_hi - edge_lo)
        elif lo < edge_lo and hi < edge_hi:
            weight = (hi - edge_lo) / (edge_hi - edge_lo)
        else:
            weight = (min(hi, edge_hi) - max(lo, edge_lo)) / (edge_hi - edge_lo)
        weights[i] = weight
    return weights


def bins_sum(values, edges, lo, hi):
    """Sum up bin values.

    Parameters
    ----------
    values: array
        values in bins
    edges: array
        bin edges
    lo: float
        lower limit
    hi: float
        upper limit

    Returns
    -------
    sum: float
        values summed from bins between lo and hi.
    """
    widths = edges[1:] - edges[:-1]
    fracs = frac_overlap_interval(edges, lo, hi)
    return np.sum(widths * values * fracs)


def bins_integrate1(values, edges, lo, hi, axis=None):
    """Integrate up bin values.

    Parameters
    ----------
    values: array
        values in bins
    edges: array
        bin edges
    lo: float
        lower limit
    hi: float
        upper limit
    axis: int | None
        summing axis

    Returns
    -------
    I: float
        integral from bins between lo and hi of values times bin center.
    """
    mids = (edges[1:] + edges[:-1]) / 2.0
    fracs = frac_overlap_interval(edges, lo, hi)
    return np.sum(mids * fracs * values, axis=axis)


def bins_integrate(values, edges, lo, hi, axis=None):
    """Integrate up bin values.

    Parameters
    ----------
    values: array
        values in bins
    edges: array
        bin edges
    lo: float
        lower limit
    hi: float
        upper limit
    axis: int | None
        summing axis

    Returns
    -------
    I: float
        values integrated from bins between lo and hi.
    """
    return np.sum(values * frac_overlap_interval(edges, lo, hi), axis=axis)


def photon_flux(unfolded_model_spectrum, energies, energy_lo, energy_hi, axis=None):
    """Compute photon flux.

    Parameters
    ----------
    unfolded_model_spectrum: array
        Model spectral density
    energies: array
        energies
    energy_lo: float
        lower limit
    energy_hi: float
        upper limit
    axis: int | None
        summing axis

    Returns
    -------
    photon_flux: float
        Photon flux in phot/cm^2/s
    """
    Nchan = len(energies) - 1
    assert unfolded_model_spectrum.shape == (Nchan,)
    assert energies.shape == (Nchan + 1,)
    integral = bins_integrate(unfolded_model_spectrum, energies, energy_lo, energy_hi, axis=axis)
    return integral / u.cm**2 / u.s


def energy_flux(unfolded_model_spectrum, energies, energy_lo, energy_hi, axis=None):
    """Compute energy flux.

    Parameters
    ----------
    unfolded_model_spectrum: array
        Model spectral density
    energies: array
        energies
    energy_lo: float
        lower limit
    energy_hi: float
        upper limit
    axis: int | None
        summing axis

    Returns
    -------
    energy_flux: float
        Energy flux in erg/cm^2/s
    """
    Nchan = len(energies) - 1
    assert unfolded_model_spectrum.shape[-1] == Nchan
    assert energies.shape == (Nchan + 1,)
    integral1 = bins_integrate1(unfolded_model_spectrum, energies, energy_lo, energy_hi, axis=axis)
    return integral1 * ((1 * u.keV).to(u.erg)) / u.cm**2 / u.s


def luminosity(
    unfolded_model_spectrum, energies, rest_energy_lo, rest_energy_hi, z, cosmo
):
    """Compute luminosity.

    Parameters
    ----------
    unfolded_model_spectrum: array
        Model spectral density
    energies: array
        energies
    rest_energy_lo: float
        lower limit
    rest_energy_hi: float
        upper limit
    z: float
        redshift
    cosmo: object
        astropy cosmology object

    Returns
    -------
    luminosity: float
        Isotropic luminosity in erg/s
    """
    Nchan = len(energies) - 1
    assert unfolded_model_spectrum.shape == (Nchan,)
    assert energies.shape == (Nchan + 1,)
    rest_energies = energies * (1 + z)
    rest_flux = energy_flux(
        unfolded_model_spectrum, rest_energies, rest_energy_lo, rest_energy_hi
    ) / (1 + z)
    DL = cosmo.luminosity_distance(z)
    return (rest_flux * (4 * np.pi * DL**2)).to(u.erg / u.s)
