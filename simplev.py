import os

import numpy as np
import scipy.stats
import ultranest
from matplotlib import pyplot as plt

import tinyxsf
import tinyxsf.flux
import tinyxsf.plot

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

def setup_model(params):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm = np.transpose(params)
    Incl = 45 + norm * 0
    Ecut = 400 + norm * 0
    scat_norm = norm * rel_scat_norm

    abs_component = absAGN(energies=energies, pars=np.transpose([NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z]), vectorized=True)

    scat_component = tinyxsf.xvec(tinyxsf.x.zpowerlw, energies=energies, pars=np.transpose([norm, PhoIndex]))
    return abs_component * norm[:,None], scat_component * scat_norm[:,None]

# define a likelihood
def loglikelihood(params, plot=False):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm = np.transpose(params)
    abs_component, scat_component = setup_model(params)

    pred_spec = abs_component + scat_component

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
param_names = ['norm', 'NH22', 'scatnorm', 'PhoIndex', 'TORsigma', 'CTKcover', 'redshift', 'bkg_norm']

outprefix = 'simplev'

# run sampler
sampler = ultranest.ReactiveNestedSampler(
    param_names, loglikelihood, prior_transform,
    log_dir=outprefix, resume=True,
    vectorized=True)
results = sampler.run(max_num_improvement_loops=0, frac_remain=0.5)

# make posterior corner plot:
sampler.plot()

# and to plot a few model posteriors:
loglikelihood(results['samples'][:10,:], plot=True)

# make pretty plots of the fit and residuals:

# Choose how many posterior samples to draw for the band (keeps plotting fast)
post = results['samples'][:400, :].copy()

# Build posterior predictive counts in source region (vectorized, like in loglikelihood)
abs_component, scat_component = setup_model(post)  # shape: (nsamp_plot, nE)
#post[:, param_names.index('NH22')] = 0.01 # make unobscured
unabs_component, _ = setup_model(post)

pred_spec = abs_component + scat_component
assert np.isfinite(pred_spec).all()

pred_counts_src_srcreg = RMF_src.apply_rmf_vectorized(
    np.einsum('i,i,ji->ji', data['ARF'], galabso, pred_spec)
)[:, data['chan_mask']] * data['src_expoarea']  # shape: (nsamp_plot, nchan_used)
pred_counts_unabs_src_srcreg = RMF_src.apply_rmf_vectorized(
    np.einsum('i,i,ji->ji', data['ARF'], galabso, unabs_component)
)[:, data['chan_mask']] * data['src_expoarea']  # shape: (nsamp_plot, nchan_used)
pred_counts_scat_src_srcreg = RMF_src.apply_rmf_vectorized(
    np.einsum('i,i,ji->ji', data['ARF'], galabso, scat_component)
)[:, data['chan_mask']] * data['src_expoarea']  # shape: (nsamp_plot, nchan_used)

pred_counts_bkg_srcreg = (
    np.einsum('j,i->ij', data['bkg_model_src_region'], post[:, param_names.index('bkg_norm')])
    * data['src_expoarea']
)
gal_fold_counts = RMF_src.apply_rmf(data['ARF'] * galabso)[data['chan_mask']]

pred_counts_arrays = [pred_counts_src_srcreg + pred_counts_bkg_srcreg, pred_counts_unabs_src_srcreg, pred_counts_scat_src_srcreg, pred_counts_src_srcreg, pred_counts_bkg_srcreg]
labels = ['Total', 'Source (intr)', 'Source (scat)', 'Source', 'Background']
colors = ['k', 'lightblue', 'lightgreen', 'orange', 'lightgray']
fig, (ax, axr) = tinyxsf.plot.plot_fit(data, pred_counts_arrays, labels, colors, rebinsig=3, rebinbnum=100, counts=True)
axr.set_xticks([0.3, 0.6, 1, 2, 3, 4, 5])
axr.set_xticklabels([0.3, 0.6, 1, 2, 3, 4, 5])
ax.legend()
plt.savefig(f'{outprefix}/plots/fit_counts.pdf')
plt.close()
fig, (ax, axr) = tinyxsf.plot.plot_fit(data, pred_counts_arrays, labels, colors, counts=True, cumulative=True, subplot='residuals')
axr.set_xticks([0.3, 0.6, 1, 2, 3, 4, 5])
axr.set_xticklabels([0.3, 0.6, 1, 2, 3, 4, 5])
ax.legend()
plt.savefig(f'{outprefix}/plots/fit_cumulative.pdf')
plt.close()
fig, (ax, axr) = tinyxsf.plot.plot_fit(data, pred_counts_arrays, labels, colors, rebinsig=3, rebinbnum=100)
axr.set_xticks([0.3, 0.6, 1, 2, 3, 4, 5])
axr.set_xticklabels([0.3, 0.6, 1, 2, 3, 4, 5])
#ax.set_ylim(0.8e-3, None)
ax.set_yscale('log')
ax.legend(loc='best')
plt.savefig(f'{outprefix}/plots/fit_convolved.pdf')
plt.close()
fig, (ax, axr) = tinyxsf.plot.plot_fit(data, pred_counts_arrays, labels, colors, rebinsig=3, rebinbnum=100, divide_vector=gal_fold_counts)
ax.set_ylabel('Counts / s / keV / cm$^2$')
#ax.set_ylim(0.5e-5, None)
ax.set_yscale('log')
axr.set_xticks([0.3, 0.6, 1, 2, 3, 4, 5])
axr.set_xticklabels([0.3, 0.6, 1, 2, 3, 4, 5])
ax.legend(loc='best')
plt.savefig(f'{outprefix}/plots/fit.pdf')
plt.close()

# compute fluxes

import astropy.units as u

soft_photon_flux_obs = tinyxsf.flux.photon_flux(galabso * abs_component, energies, 0.5, 2, axis=1)
print('soft photon flux obs:', soft_photon_flux_obs.min(), soft_photon_flux_obs.max(), soft_photon_flux_obs.mean(), soft_photon_flux_obs.std())

soft_flux_obs = tinyxsf.flux.energy_flux(galabso * abs_component, energies, 0.5, 2, axis=1)
hard_flux_obs = tinyxsf.flux.energy_flux(galabso * abs_component, energies, 2, 10, axis=1)
print('soft flux obs:', soft_flux_obs.min(), soft_flux_obs.max(), soft_flux_obs.mean(), soft_flux_obs.std())
print('hard flux obs:', hard_flux_obs.min(), hard_flux_obs.max(), hard_flux_obs.mean(), hard_flux_obs.std())
print("setting NH to zero")
soft_flux = tinyxsf.flux.energy_flux(abs_component, energies, 0.5, 2, axis=1)
hard_flux = tinyxsf.flux.energy_flux(abs_component, energies, 2, 10, axis=1)
hard_flux_restintr = hard_flux.copy()
for i, row in enumerate(post):
    z = row[6]
    hard_flux_restintr[i] = tinyxsf.flux.energy_flux(abs_component[i, :], energies, 2 / (1 + z), 10 / (1 + z))

print('soft flux:', soft_flux.min(), soft_flux.max(), soft_flux.mean(), soft_flux.std())
print('hard flux:', hard_flux.min(), hard_flux.max(), hard_flux.mean(), hard_flux.std())
print('hard rest-frame flux:', hard_flux_restintr.min(), hard_flux_restintr.max(), hard_flux_restintr.mean(), hard_flux_restintr.std())
# z, Fint, Fabs, lognorm, Gamma, logNH, fscat
redshift = post[:, param_names.index('redshift')]
norm = post[:, param_names.index('norm')]
PhoIndex = post[:, param_names.index('PhoIndex')]
NH = post[:, param_names.index('NH22')] * 1e22
#fscat = samples[:, param_names.index('scatnorm')]
np.savetxt(f'{outprefix}/chains/intrinsic_photon_flux.txt.gz', np.transpose([
    redshift, hard_flux_restintr, soft_flux_obs,
    norm, PhoIndex, NH, 
]))

# compute luminosity:
import cosmolopy.distance as cd
cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7} # standard
cosmo = cd.set_omega_k_0(cosmo)
dist = cd.quick_distance_function(cd.luminosity_distance, zmax=8, **cosmo)
L = (hard_flux_restintr * (4 * np.pi * (dist(z) * u.Mpc)**2)).to(u.erg / u.s)
print('hard luminosity:', L.min(), L.max(), L.mean(), L.std())
