import os

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import tinyxsf

# tinyxsf.x.chatter(0)
tinyxsf.x.abundance('wilm')
tinyxsf.x.cross_section('vern')

# load the spectrum, where we will consider data from 0.5 to 8 keV
data = tinyxsf.load_pha('example/179.pi', 0.5, 8)

# fetch some basic information about our spectrum
e_lo = data['e_lo']
e_hi = data['e_hi']
e_mid = (data['e_hi'] + data['e_lo']) / 2.
e_width = data['e_hi'] - data['e_lo']
energies = np.append(e_lo, e_hi[-1])
RMF_src = data['RMF_src']

chan_e = (data['chan_e_min'] + data['chan_e_max']) / 2.

# load a Table model
absAGN = tinyxsf.Table(os.path.join(os.environ.get('MODELDIR', '.'), 'uxclumpy-cutoff.fits'))

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
galabso = tinyxsf.x.TBabs(energies=energies, pars=[data['galnh']])

z = data['redshift']
# define the model parameters:
bkg_norm = 1.0
norm = 3e-7
scat_norm = norm * 0.08
PhoIndex = 2.0
TORsigma = 28.0
CTKcover = 0.1
Incl = 45.0
NH22 = 1.0
Ecut = 400


# define a likelihood
def loglikelihood(params, plot=False):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, Incl, Ecut, bkg_norm = params

    scat_norm = norm * rel_scat_norm
    # here we are taking z from the global context -- so it is fixed!
    abs_component = absAGN(energies=energies, pars=[NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z])

    scat_component = tinyxsf.x.zpowerlw(energies=energies, pars=[norm, PhoIndex])

    pred_spec = abs_component * norm + scat_component * scat_norm

    pred_counts_src_srcreg = RMF_src.apply_rmf(data['ARF'] * (galabso * pred_spec))[data['chan_mask']] * data['src_expoarea']
    pred_counts_bkg_srcreg = data['bkg_model_src_region'] * bkg_norm * data['src_expoarea']
    pred_counts_srcreg = pred_counts_src_srcreg + pred_counts_bkg_srcreg
    pred_counts_bkg_bkgreg = data['bkg_model_bkg_region'] * bkg_norm * data['bkg_expoarea']

    if plot:
        plt.figure()
        plt.legend()
        plt.plot(data['chan_e_min'], data['src_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_srcreg / (data['chan_e_max'] - data['chan_e_min']), label='src+bkg')
        plt.plot(data['chan_e_min'], pred_counts_src_srcreg / (data['chan_e_max'] - data['chan_e_min']), label='src')
        plt.plot(data['chan_e_min'], pred_counts_bkg_srcreg / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig('src_region_counts.pdf')
        plt.close()

        plt.figure()
        plt.plot(data['chan_e_min'], data['bkg_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_bkg_bkgreg / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig('bkg_region_counts.pdf')
        plt.close()

    # compute log Poisson probability
    like_srcreg = tinyxsf.logPoissonPDF(pred_counts_srcreg, data['src_region_counts'])
    like_bkgreg = tinyxsf.logPoissonPDF(pred_counts_bkg_bkgreg, data['bkg_region_counts'])
    return like_srcreg + like_bkgreg


# lets define a prior
PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)


# define the prior transform function
def prior_transform(cube):
    params = cube.copy()
    # uniform from 1e-10 to 1
    params[0] = 10**(cube[0] * -10)
    # uniform from 1e-2 to 1e2
    params[1] = 10**(cube[1] * (2 - -2) + -2)
    params[2] = 10**(cube[2] * (-1 - -5) + -5)
    # Gaussian prior
    params[3] = PhoIndex_gauss.ppf(cube[3])
    # uniform priors
    params[4] = cube[4] * (80 - 7) + 7
    params[5] = cube[5] * (0.4) + 0
    params[6] = cube[6] * 90
    params[7] = cube[7] * (400 - 300) + 300
    # log-uniform prior on the background normalisation between 0.1 and 10
    params[8] = 10**(cube[8] * (1 - -1) + -1)
    return params


# compute a likelihood:
print(loglikelihood((norm, NH22, 0.08, PhoIndex, TORsigma, CTKcover, Incl, Ecut, bkg_norm), plot=True))
