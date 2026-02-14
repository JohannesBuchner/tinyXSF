import os

import numpy as np
import scipy.stats
import ultranest
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


# define a likelihood
def loglikelihood(params, plot=False):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm = np.transpose(params)
    Incl = 45 + norm * 0
    Ecut = 400 + norm * 0
    scat_norm = norm * rel_scat_norm

    abs_component = absAGN(energies=energies, pars=np.transpose([NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z]), vectorized=True)

    scat_component = tinyxsf.xvec(tinyxsf.x.zpowerlw, energies=energies, pars=np.transpose([norm, PhoIndex]))

    pred_spec = np.einsum('ij,i->ij', abs_component, norm) + np.einsum('ij,i->ij', scat_component, scat_norm)

    pred_counts_src_srcreg = RMF_src.apply_rmf_vectorized(np.einsum('i,i,ji->ji', data['ARF'], galabso, pred_spec))[:,data['chan_mask']] * data['src_expoarea']
    pred_counts_bkg_srcreg = np.einsum('j,i->ij', data['bkg_model_src_region'], bkg_norm) * data['src_expoarea']
    pred_counts_srcreg = pred_counts_src_srcreg + pred_counts_bkg_srcreg
    pred_counts_bkg_bkgreg = np.einsum('j,i->ij', data['bkg_model_bkg_region'], bkg_norm) * data['bkg_expoarea']

    if plot:
        print(params[0])
        plt.figure()
        plt.plot(data['chan_e_min'], data['src_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='src+bkg')
        plt.plot(data['chan_e_min'], pred_counts_src_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='src')
        plt.plot(data['chan_e_min'], pred_counts_bkg_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig('src_region_counts.pdf')
        plt.close()

        plt.figure()
        plt.plot(data['chan_e_min'], data['bkg_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_bkg_bkgreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig('bkg_region_counts.pdf')
        plt.close()

    # compute log Poisson probability
    like_srcreg = tinyxsf.logPoissonPDF_vectorized(pred_counts_srcreg, data['src_region_counts'])
    like_bkgreg = tinyxsf.logPoissonPDF_vectorized(pred_counts_bkg_bkgreg, data['bkg_region_counts'])
    # combined the probabilities. If fitting multiple spectra, you would add them up here as well
    return like_srcreg + like_bkgreg


# compute a likelihood:
z = np.array([data['redshift']] * 2)
bkg_norm = np.array([1.0] * 2)
norm = np.array([3e-7] * 2)
scat_norm = norm * 0.08
PhoIndex = np.array([1.9, 2.0])
TORsigma = np.array([28.0] * 2)
CTKcover = np.array([0.1] * 2)
NH22 = np.array([1.0] * 2)
print(loglikelihood(np.transpose([norm, NH22, np.array([0.08] * 2), PhoIndex, TORsigma, CTKcover, z, bkg_norm]), plot=True))

# lets define a prior

PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)
# we are cool, we can let redshift be a free parameter informed from photo-z
z_gauss = scipy.stats.norm(data['redshift'], 0.05)


# define the prior transform function
def prior_transform(cube):
    params = cube.copy()
    # uniform from 1e-10 to 1
    params[:,0] = 10**(cube[:,0] * 10 + -10)
    # uniform from 1e-2 to 1e2
    params[:,1] = 10**(cube[:,1] * (2 - -2) + -2)
    params[:,2] = 10**(cube[:,2] * (-1 - -5) + -5)
    # Gaussian prior
    params[:,3] = PhoIndex_gauss.ppf(cube[:,3])
    # uniform priors
    params[:,4] = cube[:,4] * (80 - 7) + 7
    params[:,5] = cube[:,5] * (0.4) + 0
    # informative Gaussian prior on the redshift
    params[:,6] = z_gauss.ppf(cube[:,6])
    # log-uniform prior on the background normalisation between 0.1 and 10
    params[:,7] = 10**(cube[:,7] * (1 - -1) + -1)
    return params


# define parameter names
param_names = ['norm', 'logNH22', 'scatnorm', 'PhoIndex', 'TORsigma', 'CTKcover', 'redshift', 'bkg_norm']


# run sampler
sampler = ultranest.ReactiveNestedSampler(
    param_names, loglikelihood, prior_transform,
    log_dir='simplev', resume=True,
    vectorized=True)

# then to run:
# results = sampler.run(max_num_improvement_loops=0, frac_remain=0.5)

# and to plot a model:
# loglikelihood(results['samples'][:10,:], plot=True)

# and to plot the posterior corner plot:
# sampler.plot()
