"""

A new approach to X-ray spectral fitting with Xspec models and
simulation-based inference.

This script shows how to combine tinyXSF with the new simulation-based
inference (SBI) functionality of the latest ultranest:

1. In addition to the likelihood, we define a mock spectrum generator.
2. We train a neural posterior estimator (NPE) on simulated spectra.
   The network is amortized: it is trained across the whole prior and
   can then be applied to the observed data to predict a posterior.
3. We verify the training with a rank histogram and coverage test.
4. We use the SBI posterior to warp the prior and run the
   SBI-accelerated nested sampler (ultranest-sbi) with the exact
   Poisson likelihood.

The model is the uxclumpy-cutoff AGN table model plus a scattered power
law and a background model, as in simplev.py.

Requirements:

* latest ultranest (>= 4.6, or install from source for the sbi feature):
  pip install git+https://github.com/JohannesBuchner/UltraNest
* neural network dependencies: pip install torch torchinfo

"""
import json
import logging
import os

import corner
import numpy as np
import scipy.stats
import ultranest
import ultranest.simbase.minisbi
from matplotlib import pyplot as plt
from ultranest.simbase.minisbi.nested import (KLPTransform,
                                              get_distribution_parameters)
from ultranest.simbase.minisbi.npe import sample_posterior, train_npe
from ultranest.simbase.minisbi.plot import (parameter_coverage_test,
                                            rank_histogram)
from ultranest.simbase.minisbi.utils import make_cached_generate, make_memory

import tinyxsf
import tinyxsf.plot

tinyxsf.x.abundance('wilm')
tinyxsf.x.cross_section('vern')

# load the spectrum, where we will consider data from 0.5 to 8 keV
data = tinyxsf.load_pha('example/179.pi', 0.5, 8)

# fetch some basic information about our spectrum
energies = data['energies']
RMF_src = data['RMF_src']
Nsrc_chan = len(data['src_region_counts'])
Nbkg_chan = len(data['bkg_region_counts'])

# load a Table model
absAGN = tinyxsf.Table(os.path.join(os.environ.get('MODELDIR', '.'), 'uxclumpy-cutoff.fits'))

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
galabso = tinyxsf.x.TBabs(energies=energies, pars=[data['galnh']])

# the same model as in simplev.py, with Incl and Ecut fixed:
Incl = 45.0
Ecut = 400.0

# define the model parameters:
parameter_names = ['norm', 'NH22', 'scatnorm', 'PhoIndex', 'TORsigma', 'CTKcover', 'redshift', 'bkg_norm']
N_PARAMS = len(parameter_names)


# set up a function which computes the model components, given the parameters:
def compute_model_components(params):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm = params
    scat_norm = norm * rel_scat_norm

    abs_component = absAGN(energies=energies, pars=[NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z])
    scat_component = tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, z])

    pred_spec = abs_component * norm + scat_component * scat_norm

    pred_counts_src_srcreg = RMF_src.apply_rmf(data['ARF'] * (galabso * pred_spec))[data['chan_mask']] * data['src_expoarea']
    pred_counts_bkg_srcreg = data['bkg_model_src_region'] * bkg_norm * data['src_expoarea']
    pred_counts_bkg_bkgreg = data['bkg_model_bkg_region'] * bkg_norm * data['bkg_expoarea']
    return pred_counts_src_srcreg, pred_counts_bkg_srcreg, pred_counts_bkg_bkgreg


# define the prior transform function (from the unit hypercube to physical parameters)
PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)
# we are cool, we can let redshift be a free parameter informed from photo-z
z_gauss = scipy.stats.norm(data['redshift'], 0.05)


def prior_transform(u):
    params = u.copy()
    # log-uniform from 1e-10 to 1
    params[0] = 10**(u[0] * 10 + -10)
    # uniform from 1e-2 to 1e2
    params[1] = 10**(u[1] * (2 - -2) + -2)
    params[2] = 10**(u[2] * (-1 - -5) + -5)
    # Gaussian prior
    params[3] = PhoIndex_gauss.ppf(u[3])
    # uniform priors
    params[4] = u[4] * (80 - 7) + 7
    params[5] = u[5] * (0.4) + 0
    # informative Gaussian prior on the redshift
    params[6] = z_gauss.ppf(u[6])
    # log-uniform prior on the background normalisation between 0.1 and 10
    params[7] = 10**(u[7] * (1 - -1) + -1)
    return params


# For SBI we need a noiseless mock spectrum generator ...
def generate_mean_and_noise(theta, idx=0, seed=0):
    """Simulate a noiseless spectrum.

    It returns the expected number of counts in each channel, which is
    the mean of the Poisson distribution that generated the data.
    """
    pred_counts_src_srcreg, pred_counts_bkg_srcreg, pred_counts_bkg_bkgreg = compute_model_components(theta)
    mean = np.hstack((pred_counts_src_srcreg + pred_counts_bkg_srcreg, pred_counts_bkg_bkgreg))
    return {
        'mean': mean,
    }


# ... and a cheap noise injector (much cheaper than the model).
# We sample from the same Poisson distribution that the likelihood
# assumes, so that data generation and likelihood are mutually consistent.
def inject_noise(mean_props, rng):
    return rng.poisson(mean_props['mean'])


# the observed data vector that the neural network will condition on:
# the counts in the source region and the background region
obs_counts = np.hstack((data['src_region_counts'], data['bkg_region_counts'])).astype(float)

# the exact Poisson likelihood, for the final nested sampling run:
def loglikelihood(theta):
    pred_counts_src_srcreg, pred_counts_bkg_srcreg, pred_counts_bkg_bkgreg = compute_model_components(theta)
    like_srcreg = tinyxsf.logPoissonPDF(pred_counts_src_srcreg + pred_counts_bkg_srcreg, data['src_region_counts'])
    like_bkgreg = tinyxsf.logPoissonPDF(pred_counts_bkg_bkgreg, data['bkg_region_counts'])
    return like_srcreg + like_bkgreg


# compute a likelihood:
print(loglikelihood((3e-7, 1.0, 0.08, 2.0, 28.0, 0.1, data['redshift'], 1.0)))

# prior predictive check: how do mock spectra look across the whole prior?
plt.figure()
for i in range(20):
    u = np.random.uniform(size=N_PARAMS)
    mean_props = generate_mean_and_noise(prior_transform(u))
    plt.plot(data['chan_e_min'], mean_props['mean'][:Nsrc_chan], color='k', alpha=0.3)
plt.plot(data['chan_e_min'], obs_counts[:Nsrc_chan], 'o', color='r', mfc='none', label='data')
plt.xlabel('Channel Energy [keV]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()
plt.savefig('simplesbi-prior-predictive.pdf')
plt.close()

# Sampled datasets will be cached internally to this folder:
folder = 'simplesbi'
# optional: clear the cache whenever you change the prior or model function
# rm -rf simplesbi/

memory = make_memory(folder)
generate_noiseless_batch = make_cached_generate(memory, prior_transform, generate_mean_and_noise)

# train a neural network to predict a posterior distribution:
model = train_npe(
    # our folder
    folder=folder,
    # our (costly) model function, in batches:
    generate_noiseless_batch=generate_noiseless_batch,
    # our cheap noisy generator:
    inject_noise=inject_noise,
    # our model parameters
    n_params=N_PARAMS,

    # Parameters related to training strategy:

    # seed for randomness
    base_seed=12345,
    # how long to train: choose how long you want to wait - the higher, the
    # better. But watch the validation loss: if it becomes extremely small,
    # the network is becoming over-confident (too narrow posteriors), which
    # makes the warp below unstable. About 10 model evaluations per
    # dimension of the data vector worked well here:
    max_model_evals=10000,
    # the full history of noise-free models is kept and reused with new noise;
    # fresh_example_fraction sets the balance between freshly generated
    # models and reuse of previously generated models with new noise.
    # Because our model evaluations are cheap, we use only fresh models
    # (this avoids over-confident posteriors from over-used simulations):
    fresh_example_fraction=1.0,
    # how many new samples are generated in each batch:
    fresh_sim_batch_size=32,
    # the best neural network is kept; patience sets how many epochs
    # without a substantial improvement to wait before stopping early:
    patience=10,
    patience_min_delta=0.01,

    # parameters for the neural network architecture and learning:

    # learning rate - increase if the loss changes slowly, decrease if erratic:
    npe_lr=3e-4,
    npe_width=512,
    npe_depth=6,
    # 'cascade' is recommended - it contains both a wide and a deep network:
    layer_shape='cascade',
    npe_activation_cls='ReLU',
)

# apply the trained network to the observed spectrum to get a fast posterior:
posterior_samples_u, posterior_samples_theta = sample_posterior(
    model=model,
    observed_data=obs_counts,
    prior_transform=prior_transform,
    n_params=N_PARAMS,
    n_posterior_samples=100000,
)

# The SBI posterior may lose information or be biased. We check this:
# first with a rank histogram (should be flat):
ranks, fig, axes = rank_histogram(
    model=model,
    generate_noiseless_batch=generate_noiseless_batch,
    inject_noise=inject_noise,
    n_params=N_PARAMS,
    folder=folder,
    n_test=1000,
    seed=98765,
    param_names=parameter_names,
)
fig.savefig(folder + '/rank_histogram.pdf')
plt.close(fig)

# then with a coverage test (credible intervals should contain the truth
# in the nominal fraction of cases, i.e. follow the diagonal):
coverage, fig, axes = parameter_coverage_test(
    model=model,
    generate_noiseless_batch=generate_noiseless_batch,
    inject_noise=inject_noise,
    n_params=N_PARAMS,
    folder=folder,
    n_test=500,
    seed=11223,
    param_names=parameter_names,
)
fig.savefig(folder + '/coverage_test.pdf')
plt.close(fig)

# the neural posterior is a Kumaraswamy-Logistic product distribution on
# the unit hypercube. We query its parameters, and store them for reuse:
distribution_parameters = get_distribution_parameters(model, obs_counts)
with open(folder + '/distribution.json', 'w') as f:
    json.dump(distribution_parameters, f)

# a helper object defining the distribution, its transform and log-posterior
klp_transform = KLPTransform(**distribution_parameters)


class _State:
    last_t = None


state = _State()

# warp the prior with the NPE posterior distribution
def warped_prior_transform(t):
    # keep the last state for the likelihood
    state.last_t = t.copy()
    # [0,1]^d -> [0,1]^d with the SBI posterior warp
    u = klp_transform.transform(t).astype(float)
    # [0,1]^d -> physical parameters
    return prior_transform(u)

# correct the likelihood for the warp, so that the original prior is targeted:
def corrected_log_likelihood(theta):
    # compute our model likelihood
    logL = loglikelihood(theta)
    # add the neural posterior density at this point
    log_jac = klp_transform.log_jacobian(state.last_t)
    return logL + log_jac


# torch modifies the logging to show all messages, let's suppress that
logging.getLogger('ultranest').setLevel(logging.ERROR)

# run the SBI-accelerated sampler (ultranest-sbi):
sampler = ultranest.ReactiveNestedSampler(
    parameter_names, corrected_log_likelihood, warped_prior_transform,
    log_dir=folder, resume='overwrite')
results = sampler.run(frac_remain=0.5, max_num_improvement_loops=0)
sampler.print_results()

# make posterior corner plot, and compare it to the SBI posterior:
# the ultranest-sbi posterior is much narrower, because nested sampling
# zooms in on the information that the factorized SBI posterior lost:
figure = corner.corner(
    posterior_samples_theta,
    labels=parameter_names,
    color='steelblue',
    plot_datapoints=False, fill_contours=True, smooth=1.0,
    quantiles=[0.16, 0.5, 0.84], show_titles=True,
)
corner.corner(
    results['samples'],
    labels=parameter_names,
    fig=figure,
    color='darkorange',
    plot_datapoints=False, fill_contours=False, smooth=1.0,
    contour_kwargs={'linestyles': 'dashed'},
)
plt.savefig(folder + '/corner_comparison.pdf')
plt.close()

# make pretty plots of the fit and residuals:
# Choose how many posterior samples to draw for the band (keeps plotting fast)
post = results['samples'][:400, :]
pred_src, pred_bkg_src, pred_bkg_bkg = np.transpose(
    [compute_model_components(theta) for theta in post], (1, 0, 2))
pred_counts_arrays = [
    pred_src + pred_bkg_src,  # total in source region
    pred_src,  # source component
    pred_bkg_src,  # background model in source region
]
labels = ['Total', 'Source', 'Background']
colors = ['k', 'orange', 'lightgray']
fig, (ax, axr) = tinyxsf.plot.plot_fit(data, pred_counts_arrays, labels, colors, counts=True, rebinsig=3, rebinbnum=100)
axr.set_xticks([0.3, 0.6, 1, 2, 3, 4, 5])
axr.set_xticklabels([0.3, 0.6, 1, 2, 3, 4, 5])
ax.legend()
plt.savefig(folder + '/plots/fit_counts.pdf')
plt.close()

# print posterior summaries:
print('Posterior summary (ultranest-sbi):')
for i, name in enumerate(parameter_names):
    samp = results['samples'][:, i]
    print(f'  {name:10s}: median={np.median(samp):.4g}  '
          f'16%={np.quantile(samp, 0.16):.4g}  84%={np.quantile(samp, 0.84):.4g}')
