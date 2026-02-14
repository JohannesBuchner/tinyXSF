"""

A new approach to X-ray spectral fitting with Xspec models and
optimized nested sampling.

This script explains how to set up your model.

The idea is that you create a function which computes all model components.

"""
import corner
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from optns.sampler import OptNS
from optns.profilelike import GaussianPrior
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.730)

import tinyxsf

# tinyxsf.x.chatter(0)
tinyxsf.x.abundance('wilm')
tinyxsf.x.cross_section('vern')

# let's take a realistic example of a Chandra + NuSTAR FPMA + FPMB spectrum
# with normalisation cross-calibration uncertainty of +-0.2 dex.
# and a soft apec, a pexmon and a UXCLUMPY model, plus a background of course

# we want to make pretty plots of the fit and its components, folded and unfolded,
#  compute 2-10 keV fluxes of the components
#  compute luminosities of the intrinsic power law

filepath = '/mnt/data/daten/PostDoc2/research/agn/eROSITA/xlf/xrayspectra/NuSTARenhance/COSMOS/spectra/102/'

data_sets = {
    'Chandra': tinyxsf.load_pha(filepath + 'C.pha', 0.5, 8),
    'NuSTAR-FPMA': tinyxsf.load_pha(filepath + 'A.pha', 4, 77),
    'NuSTAR-FPMB': tinyxsf.load_pha(filepath + 'B.pha', 4, 77),
}

redshift = data_sets['Chandra']['redshift']

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
galabsos = {
    k: tinyxsf.x.TBabs(energies=data['energies'], pars=[data['galnh']])
    for k, data in data_sets.items()
}

# load a Table model
tablepath = os.path.join(os.environ.get('MODELDIR', '.'), 'uxclumpy-cutoff.fits')
import time
t0 = time.time()
print("preparing fixed table models...")
absAGN = tinyxsf.model.Table(tablepath)
print(f'took {time.time() - t0:.3f}s')
t0 = time.time()
print("preparing folded table models...")
absAGNs = {
    k: tinyxsf.model.FixedTable(
        tablepath, energies=data['energies'], redshift=redshift)
    for k, data in data_sets.items()
}
absAGN_folded = {
    k: tinyxsf.model.FixedFoldedTable(
        tablepath, energies=data['energies'], ARF=data['ARF'] * galabsos[k], RMF=data['RMF_src'], redshift=redshift, fix=dict(Ecut=400, Theta_inc=60))
    for k, data in data_sets.items()
}
print(f'took {time.time() - t0:.3f}s')
t0 = time.time()
print("preparing 1d interpolated models...")
scat_folded = {
    k: tinyxsf.model.prepare_folded_model1d(tinyxsf.x.zpowerlw, pars=[np.arange(1.0, 3.1, 0.01), redshift], energies=data['energies'], ARF=data['ARF'] * galabsos[k], RMF=data['RMF_src'])
    for k, data in data_sets.items()
}
apec_folded = {
    k: tinyxsf.model.prepare_folded_model1d(tinyxsf.x.apec, pars=[10**np.arange(-2, 1.2, 0.01), 1.0, redshift], energies=data['energies'], ARF=data['ARF'] * galabsos[k], RMF=data['RMF_src'])
    for k, data in data_sets.items()
}
print(f'took {time.time() - t0:.3f}s')
print(scat_folded['Chandra'](2.0), scat_folded['Chandra']([2.0]).shape)
assert scat_folded['Chandra'](2.0).shape == data_sets['Chandra']['chan_mask'].shape
print(apec_folded['Chandra'](2.0), apec_folded['Chandra']([2.0]).shape)
assert apec_folded['Chandra'](2.0).shape == data_sets['Chandra']['chan_mask'].shape

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
Incl = 45.0
Ecut = 400

# lets now start using optimized nested sampling.

# set up function which computes the various model components:
# the parameters are:
nonlinear_param_names = ['logNH', 'PhoIndex', 'TORsigma', 'CTKcover', 'kT']

# set up a prior transform
PhoIndex_gauss = scipy.stats.truncnorm(loc=1.95, scale=0.15, a=(1.0 - 1.95) / 0.15, b=(3.0 - 1.95) / 0.15)
def nonlinear_param_transform(cube):
    params = cube.copy()
    params[0] = cube[0] * 6 + 20    # logNH
    params[1] = PhoIndex_gauss.ppf(cube[1])
    params[2] = cube[2] * (80 - 7) + 7
    params[3] = cube[3] * (0.4) + 0
    params[4] = 10**(cube[4] * 2 - 1)  # kT
    return params

#component_names = ['pl', 'scat', 'apec']
linear_param_names = ['Nbkg', 'Npl', 'Nscat', 'Napec']
#for k in data_sets.keys():
#    for name in component_names + ['bkg']:
#        linear_param_names.append(f'norm_{name}_{k}')

bkg_deviations = 0.2
src_deviations = 0.1

Nlinear = len(linear_param_names)
Ndatasets = len(data_sets)


class LinkedPredictionPacker:
    """Map source and background spectral components to counts,

    Identical components for each dataset.

    pred_counts should look like
    pred_counts should look like
    component1-norm1: [counts_data1, 0, counts_data2, 0, counts_data3, 0]
    component2-norm2: [counts_data1, 0, counts_data2, 0, counts_data3, 0]
    component3-norm3: [counts_data1, 0, counts_data2, 0, counts_data3, 0]
    background-bkg:   [counts_srcbkg1, counts_bkgbkg1, counts_srcbkg2, counts_bkgbkg2, counts_srcbkg3, counts_bkgbkg3]
    """
    def __init__(self, data_sets, Ncomponents):
        """Initialise."""
        self.data_sets = data_sets
        self.width = 0
        self.Ncomponents = Ncomponents
        self.counts_flat = np.hstack(tuple([
            np.hstack((data['src_region_counts'], data['bkg_region_counts']))
            for k, data in data_sets.items()]))
        self.width, = self.counts_flat.shape

    def pack(self, pred_fluxes):
        # one row for each normalisation,
        pred_counts = np.zeros((self.Ncomponents, self.width))
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            for i, component_spec in enumerate(pred_fluxes[k]):
                pred_counts[i, left:left + Ndata] = component_spec
            # now look at background in the background region
            left += Ndata
            for i, component_spec in enumerate(pred_fluxes[k + '_bkg']):
                # fill in background
                pred_counts[i, left:left + Ndata] = component_spec
            left += Ndata
        return pred_counts

    def unpack(self, pred_counts):
        pred_fluxes = {}
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            pred_fluxes[k] = pred_counts[:, left:left + Ndata]
            # now look at background in the background region
            left += Ndata
            pred_fluxes[k + '_bkg'] = pred_counts[:, left:left + Ndata]
            left += Ndata
        return pred_fluxes

    def prior_prediction_producer(self, nsamples=8):
        for i in range(nsamples):
            u = np.random.uniform(size=len(statmodel.nonlinear_param_names))
            nonlinear_params = statmodel.nonlinear_param_transform(u)
            X = statmodel.compute_model_components(nonlinear_params)
            statmodel.statmodel.update_components(X)
            norms = statmodel.statmodel.norms()
            pred_counts = norms @ X.T
            yield nonlinear_params, norms, pred_counts, X

    def posterior_prediction_producer(self, samples, ypred, nsamples=8):
        for i, (params, pred_counts) in enumerate(zip(samples, ypred)):
            nonlinear_params = params[self.Ncomponents:]
            X = statmodel.compute_model_components(nonlinear_params)
            statmodel.statmodel.update_components(X)
            norms = params[:self.Ncomponents]
            assert np.allclose(pred_counts, norms @ X.T), (norms, pred_counts, norms @ X.T)
            yield nonlinear_params, norms, pred_counts, X

    def prior_predictive_check_plot(self, ax, unit='counts', nsamples=8):
        self.predictive_check_plot(ax, self.prior_prediction_producer(nsamples=nsamples), unit=unit)

    def posterior_predictive_check_plot(self, ax, samples, ypred, unit='counts'):
        self.predictive_check_plot(ax, self.posterior_prediction_producer(samples, ypred), unit=unit)

    def predictive_check_plot(self, ax, sample_infos, unit='counts', nsamples=8):
        src_factor = 1
        bkg_factor = 1
        colors = {}
        ylo = np.inf
        yhi = 0
        # now we need to unpack again:
        key_first_dataset = next(iter(self.data_sets))
        legend_entries_first_dataset = [] # data, bkg; model components, make them all black
        legend_entries_first_dataset_labels = []
        legend_entries_across_dataset = [] # take total component from each data set
        legend_entries_across_dataset_labels = []
        markers = 'osp><d^v'
        for (k, data), marker in zip(self.data_sets.items(), markers):
            if unit != 'counts':
                src_factor = 1. / data['chan_const_spec_weighting']
                bkg_factor = 1. / (data['chan_const_spec_weighting'] * data["bkg_expoarea"] / data["src_expoarea"])
            x = (data['chan_e_min'] + data['chan_e_max']) / 2.0
            l_data, = ax.plot(x, data['src_region_counts'] * src_factor, marker=marker, ls=' ', ms=2, label=f'data: {k}')
            colors[k + ' total'] = l_data.get_color()
            l_bkg_data, = ax.plot(x, data['bkg_region_counts'] * bkg_factor, marker=marker, ls=' ', ms=2, mfc='none', mec=colors[k + ' total'], label=f' bkg: {k}', alpha=0.5)
            ylo = min(ylo, np.min((0.1 + data['src_region_counts']) * src_factor))
            yhi = max(yhi, np.max(1.5 * data['src_region_counts'] * src_factor))

            if k == key_first_dataset:
                legend_entries_first_dataset += [l_data, l_bkg_data]
                legend_entries_first_dataset_labels += ['data', 'bkg']

        for i, (nonlinear_params, norms, pred_counts, X) in enumerate(sample_infos):
            left = 0
            for k, data in self.data_sets.items():
                if unit != 'counts':
                    src_factor = 1. / data['chan_const_spec_weighting']
                    bkg_factor = 1. / (data['chan_const_spec_weighting'] * data["bkg_expoarea"] / data["src_expoarea"])
                x = (data['chan_e_min'] + data['chan_e_max']) / 2.0

                Ndata = data['chan_mask'].sum()
                for j, norm in enumerate(norms):
                    if j == 0:
                        color = colors.get(k + ' total')
                        ls = '--'
                    else:
                        color = colors.get(k + ' ' + linear_param_names[j])
                        if color is None and unit != 'counts':
                            color = colors.get(linear_param_names[j])
                        ls = '-'
                    label = f'{k} {statmodel.linear_param_names[j]}' if i == 0 else None
                    l_component, = ax.plot(x, norms[j] * X[left:left + Ndata, j] * src_factor, alpha=0.5, lw=1, ls=ls, label=label, color=color)
                    colors[k + ' ' + linear_param_names[j]] = l_component.get_color()
                    colors[linear_param_names[j]] = l_component.get_color()

                    if k == key_first_dataset and i == 0:
                        legend_entries_first_dataset += [l_component]
                        legend_entries_first_dataset_labels += [statmodel.linear_param_names[j]]

                color = colors.get(k + ' total')
                label = f'{k}: total' if i == 0 else None
                l_total, = ax.plot(x, pred_counts[left:left + Ndata] * src_factor, alpha=0.5, lw=2, ls='--', color=color, label=label)
                # now look at background in the background region
                left += Ndata
                label = f'{k}: bkg' if i == 0 else None
                l_bkg, = ax.plot(x, pred_counts[left:left + Ndata] * bkg_factor, alpha=0.2, lw=2, ls=':', color=color, label=label)
                # skip plotting sub-components for background region
                left += Ndata
                if i == 0:
                    if k == key_first_dataset:
                        legend_entries_first_dataset += [l_total, l_bkg]
                        legend_entries_first_dataset_labels += ['total', 'bkg']
                    legend_entries_across_dataset.append(l_total)
                    legend_entries_across_dataset_labels.append(k)

        legend1 = ax.legend(legend_entries_first_dataset, legend_entries_first_dataset_labels, title='elements', loc='lower left')
        for h in legend1.legend_handles:
            #h.set_color('black')
            if hasattr(h, 'set_facecolor'):
                h.set_facecolor('black')
                h.set_edgecolor('black')
            #h.set_linestyle('-')
        ax.legend(legend_entries_across_dataset, legend_entries_across_dataset_labels, title='datasets', loc='lower right')
        ax.add_artist(legend1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(ylo / 10, yhi)
        ax.set_ylabel('Counts' if unit == 'counts' else 'Counts / cm$^2$ / s / keV')
        ax.set_xlabel('Energy [keV]')


# TODO, this class is not functional
class IndependentPredictionPacker:
    """Map source and background spectral components to counts,

    Independent components for each dataset.

    pred_counts should look like
    component1-norm11: [counts_data1, 0...0]
    component1-norm12: [0, counts_data2...0]
    component1-norm13: [0, ..., counts_data3]
    component2-norm21: [counts_data1, 0...0]
    component2-norm22: [0, counts_data2...0]
    component2-norm23: [0, ..., counts_data3]
    component3-norm31: [counts_data1, 0...0]
    component3-norm32: [0, counts_data2...0]
    component3-norm33: [0, ..., counts_data3]
    background-bkg:   [counts_srcbkg1, counts_bkgbkg1, counts_srcbkg2, counts_bkgbkg2, counts_srcbkg3, counts_bkgbkg3]
    """
    def __init__(self, data_sets, Ncomponents):
        """Initialise."""
        self.data_sets = data_sets
        self.Ndatasets = len(self.data_sets)
        self.Ncomponents = Ncomponents
        self.counts_flat = np.hstack(tuple([
            np.hstack((data['src_region_counts'], data['bkg_region_counts']))
            for k, data in data_sets.items()]))
        self.width, = self.counts_flat.shape

    def pack(self, pred_fluxes):
        # one row for each normalisation,
        pred_counts = np.zeros((self.Ncomponents * self.Ndatasets, self.width))
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            for i, component_spec in enumerate(pred_fluxes[k]):
                row_index = k * self.Ncomponents + i
                pred_counts[row_index, left:left + Ndata] = component_spec
            # now look at background in the background region
            left += Ndata
            for i, component_spec in enumerate(pred_fluxes[k + '_bkg']):
                row_index = k * self.Ncomponents + i
                # fill in background
                pred_counts[row_index, left:left + Ndata] = component_spec
            left += Ndata
        return pred_counts
    def unpack(self, pred_counts):
        pred_fluxes = {}
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            pred_fluxes[k] = pred_counts[:, left:left + Ndata]
            # now look at background in the background region
            left += Ndata
            pred_fluxes[k + '_bkg'] = pred_counts[:, left:left + Ndata]
            left += Ndata
        return pred_fluxes


# for some slow models and any where the params are the same across all data sets,
# it would be advantageous to call the model once with a deduplicated energy grid,
# and then distribute the results.
all_energies_with_duplicates = np.hstack([data['energies'] for k, data in data_sets.items()])
all_energies = np.unique(all_energies_with_duplicates)
#all_energies_indices = {k: np.array([np.where(all_energies == e)[0][0] for e in data['energies']])
#    for k, data in data_sets.items()}
all_energies_indices = {k: np.searchsorted(all_energies, data['energies']) for k, data in data_sets.items()}

def deduplicated_evaluation(model, pars):
    all_spec = model(energies=all_energies, pars=pars)
    all_spec_cumsum = np.concatenate([[0], all_spec.cumsum()])
    # distribute:
    results = {}
    for k, data in data_sets.items():
        energies = data['energies']
        if np.any(energies < all_energies[0]) or np.any(energies > all_energies[-1]):
            raise ValueError(f"Energy range for {k} is out of bounds of all_energies.")
        indices = all_energies_indices[k]
        # models compute the sum between energy_lo and energy_hi
        # so a wider bin needs to sum the entries between its energy_lo and energy_hi.
        indices_left = indices[:-1]
        indices_right = indices[1:]
        results[k] = all_spec_cumsum[indices_right] - all_spec_cumsum[indices_left]
    return results


# define spectral components
def compute_model_components_simple_unfolded(params):
    logNH, PhoIndex, TORsigma, CTKcover, kT = params

    # compute model components for each data set:
    apec_components = deduplicated_evaluation(tinyxsf.x.apec, pars=[kT, 1.0, redshift])

    pred_spectra = {}
    for k, data in data_sets.items():
        energies = data['energies']
        # first component: a absorbed power law
        abspl_component = absAGNs[k](energies=energies, pars=[
            10**(logNH - 22), PhoIndex, Ecut, TORsigma, CTKcover, Incl])

        # second component, a copy of the unabsorbed power law
        scat_component = tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, redshift])

        # third component, a apec model
        # apec_component = np.clip(tinyxsf.x.apec(energies=energies, pars=[kT, 1.0, redshift]), 0, None)
        apec_component = np.clip(apec_components[k], 0, None)

        background_component = data['bkg_model_src_region'] * data['src_expoarea']
        background_component_bkg_region = data['bkg_model_bkg_region'] * data['bkg_expoarea']

        pred_spectra[k] = [background_component, abspl_component, scat_component, apec_component]
        pred_spectra[k + '_bkg'] = [background_component_bkg_region]

    return pred_spectra


# fold each spectral component through the appropriate response
def compute_model_components_simple(params):
    pred_spectra = compute_model_components_simple_unfolded(params)
    pred_counts = {}
    for k, data in data_sets.items():
        pred_counts[k] = list(pred_spectra[k])
        pred_counts[k + '_bkg'] = list(pred_spectra[k + '_bkg'])
        src_spectral_components = pred_spectra[k][1:]  # skip background
        for j, src_spectral_component in enumerate(src_spectral_components):
            # now let's apply the response to each component:
            pred_counts[k][j + 1] = data['RMF_src'].apply_rmf(
                data['ARF'] * galabsos[k] * src_spectral_component)[data['chan_mask']] * data['src_expoarea']
            assert np.all(pred_counts[k][j + 1] >= 0), (k, j + 1)
    return pred_counts


# faster version, based on precomputed tables
def compute_model_components_precomputed(params):
    assert np.isfinite(params).all(), params
    logNH, PhoIndex, TORsigma, CTKcover, kT = params

    pred_counts = {}

    for k, data in data_sets.items():
        # compute model components for each data set:
        pred_counts[k] = [data['bkg_model_src_region'] * data['src_expoarea']]
        pred_counts[k + '_bkg'] = [data['bkg_model_bkg_region'] * data['bkg_expoarea']]

        # first component: a absorbed power law
        pred_counts[k].append(absAGN_folded[k](pars=[10**(logNH - 22), PhoIndex, TORsigma, CTKcover])[data['chan_mask']] * data['src_expoarea'])

        # second component, a copy of the unabsorbed power law
        pred_counts[k].append(scat_folded[k](PhoIndex)[data['chan_mask']] * data['src_expoarea'])

        # third component, a apec model
        pred_counts[k].append(apec_folded[k](kT)[data['chan_mask']] * data['src_expoarea'])
    return pred_counts


# compute_model_components = compute_model_components_precomputed
compute_model_components = compute_model_components_simple


def compute_model_components_intrinsic(params, energies):
    logNH, PhoIndex, TORsigma, CTKcover, kT = params
    pred_spectra = []
    pred_spectra.append(energies[:-1] * 0)
    pred_spectra.append(absAGN(energies=energies, pars=[0.01, PhoIndex, Ecut, TORsigma, CTKcover, Incl, redshift]))
    scat = tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, redshift])
    pred_spectra.append(scat)
    apec = np.clip(tinyxsf.x.apec(energies=energies, pars=[kT, 1.0, redshift]), 0, None)
    pred_spectra.append(apec)
    return pred_spectra


def fakeit(data_sets, norms, background=True, rng=np.random, verbose=True):
    for k, data in data_sets.items():
        counts = rng.poisson(norms @ np.array(X[k]))
        if verbose:
            print(f'  Expected counts for {k}: {np.sum(norms @ np.array(X[k]))}, actual counts: {counts.sum()}')
        # write result into the data set
        data['src_region_counts'] = counts
        if background:
            counts_bkg = rng.poisson(data['bkg_model_bkg_region'] * data['bkg_expoarea'])
            data['bkg_region_counts'] = counts_bkg
    return data_sets

for k, data in data_sets.items():
    print(k, 'expoarea:', data['src_expoarea'], data['bkg_expoarea'])
    if k.startswith('NuSTAR'):
        data['src_expoarea'] *= 50
        data['bkg_expoarea'] *= 50

# choose model parameters
X = compute_model_components([24.5, 2.0, 30.0, 0.4, 0.5])
pp = LinkedPredictionPacker(data_sets, 4)
counts_model = pp.pack(X)
# make it so that spectra have ~10000 counts each
target_counts = np.array([40, 40000, 4, 400])
norms = target_counts / counts_model.sum(axis=1)
norms[0] = 1.0

# let's compute some luminosities
print(f'norms: {norms}')

# simulate spectra and fill in the counts
print('Expected total counts:', norms @ np.sum(counts_model, axis=1))
fakeit(data_sets, norms, rng=np.random.RandomState(42))

# need a new prediction packer, because data changed
pp = LinkedPredictionPacker(data_sets, 4)

def compute_model_components_unnamed(params):
    return pp.pack(compute_model_components(params)).T

linear_param_prior_Sigma_offset = np.eye(Nlinear * Ndatasets) * 0
linear_param_prior_Sigma = np.eye(Nlinear * Ndatasets) * 0
for j in range(len(data_sets)):
    # for all data-sets, set a parameter prior:
    linear_param_prior_Sigma[j * Nlinear + 3, j * Nlinear + 3] = bkg_deviations**-2
    # across data-sets set a mutual parameter prior for each normalisation
    for k in range(j + 1, len(data_sets)):
        linear_param_prior_Sigma[j * Nlinear + 0, k * Nlinear + 0] = src_deviations**-2
        linear_param_prior_Sigma[j * Nlinear + 1, k * Nlinear + 1] = src_deviations**-2
        linear_param_prior_Sigma[j * Nlinear + 2, k * Nlinear + 2] = src_deviations**-2
    # set a prior, apply it only to the first data-set
    if j == 0:
        # -5 +- 2 for ratio of pl and scat normalisations, only on first data set
        linear_param_prior_Sigma_offset[j * Nlinear + 3, j * Nlinear + 3] = -5
        linear_param_prior_Sigma[j * Nlinear + 3, j * Nlinear + 3] = 2.0**-2


# now for the linear (normalisation) parameters:

# set up a prior log-probability density function for these linear parameters:
def linear_param_logprior_linked(params):
    lognorms = np.log(params)
    Npl = lognorms[:, linear_param_names.index('Npl')]
    Nscat = lognorms[:, linear_param_names.index('Nscat')]
    logp = np.where(Nscat > np.log(0.1) + Npl, -np.inf, 0)
    return logp

def linear_param_logprior_independent(params):
    assert np.all(params > 0)
    lognorms = np.log(params.reshape((-1, Nlinear, len(data_sets))))
    Npl = lognorms[:, linear_param_names.index('Npl'), :]
    Nscat = lognorms[:, linear_param_names.index('Nscat'), :]
    #Napec = lognorms[:, component_names.index('apec'), :]
    #Nbkg = lognorms[:, component_names.index('bkg'), :]
    logp = np.where(Nscat > np.log(0.1) + Npl, -np.inf, 0)
    return logp

# we should be able to handle two cases:
#   the model normalisations are identical across data sets (LinkedPredictionPacker)
#      in that case, we have only few linear parameters
#   the model normalisations are free parameters in each data set (IndependentPredictionPacker)
#      in that case, we have many linear parameters, and we add a Gaussian prior on the lognorms deviations across data sets
#   in both cases, the variation across normalisations can also be a prior

# put a prior on the ratio of Npl and Nscat
ratio_mutual_priors = [2, 1, -5, 2.0]

# lognorms_prior = pp.create_linear_param_prior(ratio_mutual_priors)
#linear_param_prior_Sigma_offset, linear_param_prior_Sigma
#lognorms_prior = GaussianPrior(linear_param_prior_Sigma_offset, linear_param_prior_Sigma)




# create OptNS object, and give it all of these ingredients,
# as well as our data

# we will need some glue between OptNS and our dictionaries
statmodel = OptNS(
    linear_param_names, nonlinear_param_names, compute_model_components_unnamed,
    nonlinear_param_transform, linear_param_logprior_linked,
    pp.counts_flat, positive=True)
#statmodel.statmodel.minimize_kwargs['options']['ftol'] = 1e-2

# prior predictive checks:
fig = plt.figure(figsize=(15, 4))
pp.prior_predictive_check_plot(fig.gca())
#plt.legend(ncol=4)
plt.savefig('multispecopt-priorpc-counts.pdf')
plt.close()

fig = plt.figure(figsize=(15, 4))
pp.prior_predictive_check_plot(fig.gca(), unit='area')
#plt.legend(ncol=4)
plt.savefig('multispecopt-priorpc.pdf')
plt.close()
print("starting benchmark...")
import time, tqdm
t0 = time.time()
for i in tqdm.trange(1000):
    u = np.random.uniform(size=len(statmodel.nonlinear_param_names))
    nonlinear_params = statmodel.nonlinear_param_transform(u)
    assert np.isfinite(nonlinear_params).all()
    X = statmodel.compute_model_components(nonlinear_params)
    assert np.isfinite(X).all()
    statmodel.statmodel.update_components(X)
    norms = statmodel.statmodel.norms()
    assert np.isfinite(norms).all()
    pred_counts = norms @ X.T
print('Duration:', (time.time() - t0) / 1000)

# create a UltraNest sampler from this. You can pass additional arguments like here:
optsampler = statmodel.ReactiveNestedSampler(
    log_dir='multispecoptjit', resume=True)
# run the UltraNest optimized sampler on the nonlinear parameter space:
optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
optsampler.print_results()
optsampler.plot()

# now for postprocessing the results, we want to get the full posterior:
# this samples up to 1000 normalisations for each nonlinear posterior sample:
fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'][:200], 40)
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
plt.savefig('multispecopt-corner.pdf')
plt.close()

# to obtain equally weighted samples, we resample
# this respects the effective sample size. If you get too few samples here,
# crank up the number just above.
samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
print(f'Obtained {len(samples)} equally weighted posterior samples')

# posterior predictive checks:
fig = plt.figure(figsize=(15, 4))
pp.posterior_predictive_check_plot(fig.gca(), samples[:40], y_pred_samples[:40])
plt.savefig('multispecopt-ppc-counts.pdf')
plt.close()

fig = plt.figure(figsize=(15, 4))
pp.posterior_predictive_check_plot(fig.gca(), samples[:40], y_pred_samples[:40], unit='area')
plt.savefig('multispecopt-ppc.pdf')
plt.close()

from tinyxsf.flux import luminosity, energy_flux
import astropy.units as u

luminosities = []
energy_fluxes = []
luminosities2 = []
energy_fluxes2 = []
for i, (params, pred_counts) in enumerate(zip(samples[:40], y_pred_samples[:40])):
    norms = params[:Nlinear]
    nonlinear_params = params[Nlinear:]
    X = compute_model_components_intrinsic(nonlinear_params, data_sets['Chandra']['energies'])
    X2 = compute_model_components_intrinsic(nonlinear_params, data_sets['NuSTAR-FPMA']['energies'])

    energy_fluxes.append([energy_flux(Ni * Xi, data_sets['Chandra']['energies'], 2, 8) / (u.erg/u.s/u.cm**2) for Ni, Xi in zip(norms, X)])
    luminosities.append([luminosity(Ni * Xi, data_sets['Chandra']['energies'], 2, 10, z=redshift, cosmo=cosmo) / (u.erg/u.s) for Ni, Xi in zip(norms, X)])

    energy_fluxes2.append([energy_flux(Ni * Xi, data_sets['NuSTAR-FPMA']['energies'], 2, 8) / (u.erg/u.s/u.cm**2) for Ni, Xi in zip(norms, X2)])
    luminosities2.append([luminosity(Ni * Xi, data_sets['NuSTAR-FPMA']['energies'], 2, 10, z=redshift, cosmo=cosmo) / (u.erg/u.s) for Ni, Xi in zip(norms, X2)])

luminosities = np.array(luminosities)
luminosities2 = np.array(luminosities2)
energy_fluxes = np.array(energy_fluxes)
energy_fluxes2 = np.array(energy_fluxes2)
print("Luminosities[erg/s]:")
print(np.mean(luminosities.sum(axis=1)), np.std(luminosities.sum(axis=1)))
print(np.mean(luminosities, axis=0))
print(np.std(luminosities, axis=0))
print(np.mean(luminosities2, axis=0))
print(np.std(luminosities2, axis=0))
print()

print("Energy fluxes[erg/s/cm^2]:")
print(np.mean(energy_fluxes.sum(axis=1)), np.std(energy_fluxes.sum(axis=1)))
print(np.mean(energy_fluxes, axis=0))
print(np.std(energy_fluxes, axis=0))
print(np.mean(energy_fluxes2, axis=0))
print(np.std(energy_fluxes2, axis=0))
print()







