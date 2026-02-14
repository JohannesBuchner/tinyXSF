import numpy as np
import tinyxsf
from astropy import units as u
from tinyxsf.flux import energy_flux, photon_flux, luminosity, frac_overlap_interval
import os
from astropy.cosmology import LambdaCDM
from numpy.testing import assert_allclose

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.730)

# tinyxsf.x.chatter(0)
tinyxsf.x.abundance('wilm')
tinyxsf.x.cross_section('vern')

# load the spectrum, where we will consider data from 0.5 to 8 keV
data = tinyxsf.load_pha(os.path.join(os.path.dirname(__file__), '../example/179.pi'), 0.5, 8)
energies = np.append(data['e_lo'], data['e_hi'][-1])
chan_e = np.append(data['chan_e_min'], data['chan_e_max'][-1])

print(data['src_expoarea'])

def test_frac_overlap_interval():
    assert_allclose(frac_overlap_interval(np.asarray([0, 1, 2, 3]), 1, 2), [0, 1, 0])
    assert_allclose(frac_overlap_interval(np.asarray([0, 1, 2, 3]), 1.5, 2.5), [0, 0.5, 0.5])
    assert_allclose(frac_overlap_interval(np.asarray([0.1, 1.1, 2.1, 3.1]), 1.5, 2.5), [0, 0.6, 0.4])
    assert_allclose(frac_overlap_interval(np.asarray([0.1, 0.2, 0.3, 0.4]), 0.1, 0.12), [0.2, 0, 0])
    assert_allclose(frac_overlap_interval(np.asarray([0.1, 0.2, 0.3, 0.4]), 0.28, 0.3), [0, 0.2, 0])
    assert_allclose(frac_overlap_interval(np.asarray([0.1, 0.2, 0.3, 0.4]), 0.38, 0.4), [0, 0, 0.2])

def test_powerlaw_fluxes():
    pl = tinyxsf.x.zpowerlw(energies=energies, pars=[1, 0])
    # print(photon_flux(pl, energies, 2.001, 2.002), 1.6014e-12 * u.erg/u.cm**2/u.s)
    assert np.isclose(photon_flux(pl, energies, 2.001, 2.002), 0.00049938 / u.cm**2/u.s, atol=1e-14)
    assert np.isclose(energy_flux(pl, energies, 2.001, 2.002), 1.6014e-12 * u.erg/u.cm**2/u.s, atol=1e-14)
    assert np.isclose(energy_flux(pl, energies, 2, 8), 9.6132e-09 * u.erg/u.cm**2/u.s, atol=1e-13)

    zpl0 = tinyxsf.x.zpowerlw(energies=energies, pars=[1, 0.01])
    assert np.isclose(energy_flux(zpl0, energies, 2, 8), 9.518e-09 * u.erg/u.cm**2/u.s, atol=1e-13)
    assert np.isclose(luminosity(zpl0, energies, 2, 8, 0.01, cosmo=cosmo), 2.0971e+45 * u.erg/u.s, rtol=0.04)

    zpl = tinyxsf.x.zpowerlw(energies=energies, pars=[1, 0.5])
    assert np.isclose(energy_flux(zpl, energies, 2, 8), 6.4088e-09 * u.erg/u.cm**2/u.s, atol=1e-11)
    assert np.isclose(luminosity(zpl, energies, 2, 10, 0.5, cosmo=cosmo), 5.5941e+48 * u.erg/u.s, rtol=0.04)

    zpl3 = tinyxsf.x.zpowerlw(energies=energies, pars=[3, 0.5])
    assert np.isclose(energy_flux(zpl3, energies, 2, 8), 1.7802e-10 * u.erg/u.cm**2/u.s, atol=1e-11)
    assert np.isclose(luminosity(zpl3, energies, 2, 10, 0.5, cosmo=cosmo), 2.7971e+47 * u.erg/u.s, rtol=0.04)
    assert np.isclose(photon_flux(zpl3, energies, 2, 8), 0.034722 / u.cm**2/u.s, atol=1e-14)

