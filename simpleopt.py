"""

A new approach to X-ray spectral fitting with Xspec models and 
optimized nested sampling.

This script explains how to set up your model.

The idea is that you create a function which computes all model components.

"""
import os
import corner
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from optns.sampler import OptNS

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

# z = data['redshift']
# define the model parameters:
#bkg_norm = 1.0
#norm = 3e-7
#scat_norm = norm * 0.08
#PhoIndex = 2.0
#TORsigma = 28.0
#CTKcover = 0.1
Incl = 45.0
#NH22 = 1.0
Ecut = 400

Nsrc_chan = len(data['src_region_counts'])
Nbkg_chan = len(data['bkg_region_counts'])
counts_flat = np.hstack((data['src_region_counts'], data['bkg_region_counts']))

# lets now start using optimized nested sampling.

# set up function which computes the various model components:
# the parameters are:
nonlinear_param_names = ['logNH', 'PhoIndex', 'TORsigma', 'CTKcover', 'redshift']


def compute_model_components(params):
    logNH, PhoIndex, TORsigma, CTKcover, z = params

    # first component: a absorbed power law
    NH22 = 10**(logNH - 22)
    abs_component = absAGN(energies=energies, pars=[NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z])

    # second component, a copy of the unabsorbed power law
    scat = tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, z])

    # lets compute the detected counts:
    pred_counts = np.zeros((3, Nsrc_chan + Nbkg_chan))
    # the two folded source components in the source region:
    pred_counts[0, :Nsrc_chan] = RMF_src.apply_rmf(data['ARF'] * galabso * abs_component)[data['chan_mask']] * data['src_expoarea']
    pred_counts[1, :Nsrc_chan] = RMF_src.apply_rmf(data['ARF'] * galabso * scat)[data['chan_mask']] * data['src_expoarea']
    # the unfolded background components in the source region
    pred_counts[2, :Nsrc_chan] = data['bkg_model_src_region'] * data['src_expoarea']
    # the unfolded background components in the background region
    pred_counts[2, Nsrc_chan:] = data['bkg_model_bkg_region'] * data['bkg_expoarea']
    # notice how the first three components do not affect the background:
    # pred_counts[0:3, Nsrc_chan:] = 0  # they remain zero
    assert (pred_counts[0] > 0).any(), (params, abs_component)
    assert (pred_counts[1] > 0).any(), (params, scat)
    assert (pred_counts[2] > 0).any(), (params,)

    return pred_counts.T


# set up a prior transform for these nonlinear parameters
PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)
# we are cool, we can let redshift be a free parameter informed from photo-z
z_gauss = scipy.stats.norm(data['redshift'], 0.05)


def nonlinear_param_transform(cube):
    params = cube.copy()
    params[0] = cube[0] * 4 + 20    # logNH
    params[1] = PhoIndex_gauss.ppf(cube[1])
    params[2] = cube[2] * (80 - 7) + 7
    params[3] = cube[3] * (0.4) + 0
    # informative Gaussian prior on the redshift
    params[4] = z_gauss.ppf(cube[4])
    return params


# now for the linear (normalisation) parameters:
linear_param_names = ['Nsrc', 'Nscat', 'Nbkg']

# set up a prior log-probability density function for these linear parameters:


def linear_param_logprior(params):
    assert np.all(params > 0)
    Nsrc, Nscat, Nbkg = params.transpose()
    # a log-uniform prior on the source luminosity
    logp = -np.log(Nsrc)
    # a log-uniform prior on the relative scattering normalisation.
    # logp = -np.log(Nscat / Nsrc)
    assert np.isfinite(logp).all(), logp
    # limits:
    logp[Nscat > 0.1 * Nsrc] = -np.inf
    return logp


# create OptNS object, and give it all of these ingredients,
# as well as our data
statmodel = OptNS(
    linear_param_names, nonlinear_param_names, compute_model_components,
    nonlinear_param_transform, linear_param_logprior,
    counts_flat, positive=True)

# prior predictive checks:
fig = plt.figure(figsize=(15, 4))
statmodel.prior_predictive_check_plot(fig.gca())
plt.legend()
plt.ylim(0.1, counts_flat.max() * 1.1)
plt.yscale('log')
plt.savefig('simpleopt-ppc.pdf')
plt.close()

# create a UltraNest sampler from this. You can pass additional arguments like here:
optsampler = statmodel.ReactiveNestedSampler(
    log_dir='simpleopt', resume=True)
# run the UltraNest optimized sampler on the nonlinear parameter space:
optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
optsampler.print_results()
optsampler.plot()

# now for postprocessing the results, we want to get the full posterior:
# this samples up to 1000 normalisations for each nonlinear posterior sample:
fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'][:400], 100)
print(f'Obtained {len(fullsamples)} weighted posterior samples')

print('weights:', weights, np.nanmin(weights), np.nanmax(weights), np.mean(weights))
# make a corner plot:
mask = weights > 1e-6 * np.nanmax(weights)
fullsamples_selected = fullsamples[mask,:]
fullsamples_selected[:, :len(linear_param_names)] = np.log10(fullsamples_selected[:, :len(linear_param_names)])

print(f'Obtained {mask.sum()} with not minuscule weight.')
fig = corner.corner(
    fullsamples_selected, weights=weights[mask],
    labels=linear_param_names + nonlinear_param_names,
    show_titles=True, quiet=True,
    plot_datapoints=False, plot_density=False,
    levels=[0.9973, 0.9545, 0.6827, 0.3934], quantiles=[0.15866, 0.5, 0.8413],
    contour_kwargs=dict(linestyles=['-','-.',':','--'], colors=['navy','navy','navy','purple']),
    color='purple'
)
plt.savefig('simpleopt-corner.pdf')
plt.close()

# to obtain equally weighted samples, we resample
# this respects the effective sample size. If you get too few samples here,
# crank up the number just above.
samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
print(f'Obtained {len(samples)} equally weighted posterior samples')


# prior predictive checks:
fig = plt.figure(figsize=(15, 10))
statmodel.posterior_predictive_check_plot(fig.gca(), samples[:100])
plt.legend()
plt.ylim(0.1, counts_flat.max() * 1.1)
plt.yscale('log')
plt.savefig('simpleopt-postpc.pdf')
plt.close()

