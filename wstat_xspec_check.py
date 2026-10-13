"""Cross-validate tinyxsf.wstat against the wstat statistic of XSPEC.

pyXSPEC and xspec_models_cxc cannot be imported into the same process
(their XSPEC model libraries clash), so this script runs in two phases:
`fit` fits the example spectrum in XSPEC (statistic cstat, which uses
wstat when a background file is loaded) and stores the best fit
parameters and statistic; `check` evaluates tinyxsf.wstat at those
parameters with the same response folding and compares.

Run inside the example/ directory with a HEASOFT environment (HEADAS,
PYTHONPATH) for the fit phase:

    python wstat_xspec_check.py fit
    python wstat_xspec_check.py check
"""
import json
import os
import sys

RESULT_FILE = 'wstat_xspec_check.json'

if len(sys.argv) != 2 or sys.argv[1] not in ('fit', 'check'):
    sys.exit(__doc__)

mode = sys.argv[1]

if mode == 'fit':
    import xspec

    xspec.Xset.chatter = 0
    spec = xspec.Spectrum('179.pi')
    xspec.Fit.statMethod = 'cstat'  # with a background file this uses wstat
    model = xspec.Model('powerlaw')
    xspec.Fit.perform()
    results = dict(
        statistic=xspec.Fit.statistic,
        dof=xspec.Fit.dof,
        gamma=model.powerlaw.PhoIndex.values[0],
        norm=model.powerlaw.norm.values[0],
    )
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f)
    print('XSPEC: statistic={statistic:.6f} dof={dof} '
          '(Gamma={gamma:.6f}, norm={norm:.6g})'.format(**results))
else:
    import numpy as np
    import scipy.optimize

    import tinyxsf

    with open(RESULT_FILE) as f:
        results = json.load(f)
    gamma, norm = results['gamma'], results['norm']

    data = tinyxsf.load_pha('179.pi', 0, 1e6)
    assert data['Nchan'] == 1024, data['Nchan']
    energies = np.append(data['e_lo'], data['e_hi'][-1])

    def model_counts(gamma_, norm_):
        # keep the optimizer exploration in a range where the model is finite
        gamma_ = np.clip(gamma_, -20, 20)
        # xspec_models_cxc in this version has no norm parameter:
        # the models return the spectrum for norm=1, multiply separately
        spec_model = tinyxsf.x.powerlaw(energies=energies, pars=[gamma_]) * norm_
        return data['RMF_src'].apply_rmf(
            data['ARF'] * spec_model)[data['chan_mask']] * data['src_expoarea']

    def wstat_of_pars(logpars):
        return tinyxsf.wstat(model=model_counts(*np.exp(logpars)),
                             src_counts=data['src_region_counts'],
                             bkg_counts=data['bkg_region_counts'],
                             src_expoarea=data['src_expoarea'],
                             bkg_expoarea=data['bkg_expoarea'],
                             backscale_ratio=data['backscale_ratio'])

    w = wstat_of_pars(np.log([gamma, norm]))
    res = scipy.optimize.minimize(
        wstat_of_pars, np.log([gamma, norm]), method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8})
    print('XSPEC:   wstat={statistic:.6f} (dof={dof})'.format(**results))
    print('tinyxsf: wstat={:.6f} at the XSPEC best fit'.format(w))
    print('tinyxsf: wstat={:.6f} at own minimum '
          '(Gamma={:.6f}, norm={:.6g})'.format(
              res.fun, np.exp(res.x[0]), np.exp(res.x[1])))
    print('relative difference at XSPEC best fit: {:.2e}'.format(
        abs(w - results['statistic']) / results['statistic']))
