"""Functionality for loading data."""
import os
from functools import cache

import astropy.io.fits as pyfits
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from .response import ARF, RMF, MockARF


@cache
def get_ARF(arf_filename):
    """Read ancillary response file.

    Avoids building a new object for the same file with caching.

    Parameters
    ----------
    arf_filename: str
        filename

    Returns
    -------
    ARF: object
        ARF object
    """
    return ARF(arf_filename)


@cache
def get_RMF(rmf_filename):
    """Read response matrix file.

    Avoids building a new object for the same file with caching.

    Parameters
    ----------
    rmf_filename: str
        filename

    Returns
    -------
    RMF: object
        RMF object
    """
    return RMF(rmf_filename)


def load_pha(filename, elo, ehi, load_absorption=True, z=None, require_background=True):
    """Load PHA file.

    Parameters
    ----------
    filename: str
        file name of the PHA-style spectrum
    elo: float
        lowest energy channel to consider
    ehi: float
        highest energy channel to consider
    load_absorption: bool
        whether to try to load the <filename>.nh file
    z: float or None
        if given, set data['redshift'] to z.
        Otherwise try to load the <filename>.z file.
    require_background: bool
        whether to fail if no corresponding background file is associated
        with the data.

    Returns
    -------
    data: dict
        All information about the observation.
    """
    path = os.path.dirname(filename)
    a = pyfits.open(filename)
    header = a["SPECTRUM"].header
    exposure = header["EXPOSURE"]
    backscal = header["BACKSCAL"]
    areascal = header["AREASCAL"]
    backfile = os.path.join(path, header["BACKFILE"])
    rmffile = os.path.join(path, header["RESPFILE"])
    arffile = os.path.join(path, header["ANCRFILE"])
    channels = a["SPECTRUM"].data["CHANNEL"]

    b = pyfits.open(backfile)
    bheader = b["SPECTRUM"].header
    bexposure = bheader["EXPOSURE"]
    bbackscal = bheader["BACKSCAL"]
    bareascal = bheader["AREASCAL"]
    assert (
        "RESPFILE" not in bheader or bheader["RESPFILE"] == header["RESPFILE"]
    ), "background must have same RMF"
    assert (
        "ANCRFILE" not in bheader or bheader["ANCRFILE"] == header["ANCRFILE"]
    ), "background must have same ARF"

    ebounds = pyfits.getdata(rmffile, "EBOUNDS")
    chan_e_min = ebounds["E_MIN"]
    chan_e_max = ebounds["E_MAX"]
    mask = np.logical_and(chan_e_min > elo, chan_e_max < ehi)
    armf = get_RMF(rmffile)
    m = armf.get_dense_matrix()
    Nflux, Nchan = m.shape

    assert (Nflux,) == armf.energ_lo.shape == armf.energ_hi.shape

    if header["ANCRFILE"].strip() == 'NONE':
        aarf = MockARF(armf)
    else:
        aarf = get_ARF(arffile)

    assert (Nflux,) == aarf.e_low.shape == aarf.e_high.shape
    assert len(channels) == Nchan, (len(channels), Nchan)

    strip_mask = armf.strip(mask)
    aarf.strip(strip_mask)
    assert armf.energ_lo.shape == armf.energ_hi.shape == aarf.e_low.shape == aarf.e_high.shape

    # assert np.allclose(channels, np.arange(Nchan)+1), (channels, Nchan)
    fcounts = a["SPECTRUM"].data["COUNTS"]
    assert (Nchan,) == fcounts.shape, (fcounts.shape, Nchan)
    counts = fcounts.astype(int)
    assert (counts == fcounts).all()

    bchannels = b["SPECTRUM"].data["CHANNEL"]
    # assert np.allclose(bchannels, np.arange(Nchan)+1), (bchannels, Nchan)
    assert len(bchannels) == Nchan, (len(bchannels), Nchan)
    bfcounts = b["SPECTRUM"].data["COUNTS"]
    assert (Nchan,) == bfcounts.shape, (bfcounts.shape, Nchan)
    bcounts = bfcounts.astype(int)
    assert (bcounts == bfcounts).all()

    data = dict(
        Nflux=Nflux,
        Nchan=mask.sum(),
        src_region_counts=counts[mask],
        bkg_region_counts=bcounts[mask],
        chan_mask=mask,
        RMF_src=armf,
        RMF=np.array(m[:, mask][strip_mask,:]),
        ARF=np.array(aarf.specresp, dtype=float),
        e_lo=np.array(aarf.e_low),
        e_hi=np.array(aarf.e_high),
        e_delta=np.array(aarf.e_high - aarf.e_low),
        energies=np.append(aarf.e_low, aarf.e_high[-1]),
        chan_e_min=chan_e_min[mask],
        chan_e_max=chan_e_max[mask],
        src_expo=exposure,
        bkg_expo=bexposure,
        src_expoarea=exposure * areascal,
        bkg_expoarea=bexposure * bareascal,
        src_to_bkg_ratio=areascal / bareascal * backscal / bbackscal * exposure / bexposure,
    )
    data["chan_const_spec_weighting"] = data["RMF_src"].apply_rmf(
        data["ARF"] * data['e_lo']**-1.4)[data['chan_mask']] * data['src_expoarea']

    # data["chan_const_spec_weighting"] = np.dot(data["ARF"], data["RMF"]) * data["src_expoarea"]

    if os.path.exists(backfile + "_model.fits"):
        bkg_model = pyfits.getdata(backfile + "_model.fits", "SPECTRA")
        data["bkg_model_bkg_region"] = np.array(bkg_model[0]["INTPSPEC"][mask])
        data["bkg_model_src_region"] = np.array(bkg_model[1]["INTPSPEC"][mask])
    if os.path.exists(filename + ".nh") and load_absorption:
        data["galnh"] = float(np.loadtxt(filename + ".nh") / 1e22)
    if z is not None:
        data["redshift"] = z
        data["e_mid_restframe"] = (data["e_hi"] + data["e_lo"]) / 2 * (1 + z)
        data["luminosity_distance"] = cosmo.luminosity_distance(z).to(u.cm)
    elif os.path.exists(filename + ".z"):
        z = float(np.loadtxt(filename + ".z"))
        data["redshift"] = z
        data["e_mid_restframe"] = (data["e_hi"] + data["e_lo"]) / 2 * (1 + z)
        data["luminosity_distance"] = cosmo.luminosity_distance(z).to(u.cm)
    else:
        z = 0.0

    return data
