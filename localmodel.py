#!/usr/bin/env python
"""Using a compiled XSPEC local model (relxill) with tinyXSF.

"Local models" are compiled model packages that are not part of XSPEC's
model.dat (and thus not in a standard xspec_models_cxc build), such as
relxill compiled from source into librelxill.so. The xspec_models_cxc in
the heasoft environment was built with relxill included, so the model is
available as tinyxsf.x.relxill. Two things are required:

1) Preload the model's shared library with ctypes (RTLD_GLOBAL) BEFORE
   importing tinyxsf (which imports xspec_models_cxc): the compiled
   interface references the model function (lmodrelxill), but does not
   link the library, and Python loads extension modules with RTLD_NOW.
   Without the preload, `import tinyxsf` merely warns "Unable to import
   compiled XSPEC models" and no model is available at all.

2) The norm is the implicit last XSPEC parameter -- it is not an argument
   of the model function and not in the parameter list either. Multiply
   the returned spectrum by it yourself.

The available parameters of a model, their order, defaults, units and
allowed ranges are listed with::

    tinyxsf.x.info('relxill').parameters

The pars list passed to the model must follow exactly that order. If your
xspec_models_cxc build does not know the model, look up the parameter
order in <model directory>/lmodel.dat instead, and either build the model
into xspec_models_cxc or evaluate the shared library directly through
ctypes (see the tinyxsf-local-models skill, templates/relxill_ctypes.py).

Run in the heasoft environment (so that LD_LIBRARY_PATH resolves the
libraries that the model library links against)::

    MAMBA_ROOT_PREFIX=/home/user/micromamba mamba run -n heasoft python localmodel.py

This example plots the unfolded (not folded through any instrument
response) relxill spectrum of a Gamma=2 powerlaw AGN for spin a=0 and
a=0.998, each normalized to a 2-10 keV flux of 1e-12 erg/s/cm^2, and
writes it to localmodel_relxill_spins.pdf/.png.
"""
import ctypes

# relxill installed as an XSPEC local model (compiled in place; its tables
# are read from the path baked in at compile time)
RELXILL_DIR = '/home/user/Downloads/xmmproposal/relxill'

# preload the compiled relxill model library -- see module docstring
ctypes.CDLL(RELXILL_DIR + '/librelxill.so', mode=ctypes.RTLD_GLOBAL)

import numpy as np
import astropy.units as u
import tinyxsf
import tinyxsf.flux
from matplotlib import pyplot as plt

tinyxsf.x.chatter(0)

# list the available parameters -- see module docstring
info = tinyxsf.x.info('relxill')
print(f'model "{info.name}", {len(info.parameters)} parameters '
      '(+ the implicit norm), in the order the model expects them:')
for i, p in enumerate(info.parameters):
    limits = '' if p.hardmin is None else f' range [{p.hardmin}, {p.hardmax}]'
    print(f'  {i:2d}: {p.name:9s} default={p.default:<8g} units={p.units or "-"}{limits}')


# relxill parameter vector in the order printed above, for an AGN whose
# disk is illuminated by a Gamma=2 powerlaw:
def relxill_pars(a):
    return [
        3.0,    # Index1: emissivity profile index (inner disk)
        3.0,    # Index2: emissivity profile index (outer disk)
        15.0,   # Rbr: break radius of the emissivity profile (r_g)
        a,      # a: dimensionless black hole spin
        30.0,   # Incl: disk inclination (deg)
        -1.0,   # Rin: inner disk radius (< 0: ISCO)
        400.0,  # Rout: outer disk radius (r_g)
        0.1,    # z: redshift
        2.0,    # gamma: photon index Gamma of the powerlaw
        3.0,    # logxi: disk ionization
        1.0,    # Afe: iron abundance (solar)
        300.0,  # Ecut: powerlaw high-energy cutoff (keV)
        1.0,    # refl_frac: reflection strength
    ]


a_values = [0.0, 0.998]

# N+1 bin edges; the model returns one flux value per bin
energies = np.logspace(np.log10(0.5), np.log10(30.0), 3001)
delta_e = energies[1:] - energies[:-1]
e_mid = 0.5 * (energies[1:] + energies[:-1])

# evaluate for both spins: shape (len(a_values), len(energies) - 1),
# photon spectrum per bin in phot/cm^2/s, for norm=1
specs = np.array([
    tinyxsf.x.relxill(energies=energies, pars=relxill_pars(a))
    for a in a_values
])

# the norm is not a model parameter -- apply it externally: normalize both
# spins to the same observed-frame 2-10 keV flux. The flux functions take
# the unfolded model spectrum (photons/cm^2/s per bin), before ARF/RMF
# folding, like xagnfitter.py and tests/test_lum.py use them.
flux_target = 1e-12  # erg/s/cm^2
flux_per_norm = tinyxsf.flux.energy_flux(specs, energies, 2, 10, axis=1)
norms = flux_target / flux_per_norm.to(u.erg / u.s / u.cm**2).value
specs = specs * norms[:, None]
print('norms:', norms)
print('F(2-10 keV):', tinyxsf.flux.energy_flux(specs, energies, 2, 10, axis=1))

# plot the unfolded (not folded) spectrum as nu*F_nu = E^2 * dN/dE
KEV2ERG = 1.60218e-9
enuf = specs / delta_e * e_mid**2 * KEV2ERG  # erg/s/cm^2

fig, ax = plt.subplots(figsize=(8, 5))
for a, y in zip(a_values, enuf):
    ax.loglog(e_mid, y, lw=1.6, label=f'$a = {a}$')
ax.set_xlabel('E [keV]')
ax.set_ylabel(r'$\nu F_\nu$ [erg s$^{-1}$ cm$^{-2}$]')
ax.set_title('relxill, unfolded (no ARF/RMF folding)')
ax.text(0.02, 0.96,
        r'$\Gamma=2$, $\log\xi=3$, $A_{\rm Fe}=1$, $E_{\rm cut}=300$ keV,'
        r' $i=30°$, $z=0.1$',
        transform=ax.transAxes, va='top', fontsize=9, color='0.3')
ax.legend(loc='lower left', frameon=False)
fig.tight_layout()
fig.savefig('localmodel_relxill_spins.pdf')
fig.savefig('localmodel_relxill_spins.png', dpi=150)
print('written localmodel_relxill_spins.pdf/.png')
