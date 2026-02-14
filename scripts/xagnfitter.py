import os

import numpy as np
import scipy.stats
import ultranest
from matplotlib import pyplot as plt

import tinyxsf
import tinyxsf.flux
import sys
import astropy.units as u

# tinyxsf.x.chatter(0)
tinyxsf.x.abundance('wilm')
tinyxsf.x.cross_section('vern')

# load the spectrum, where we will consider data from 0.5 to 8 keV
data = tinyxsf.load_pha(sys.argv[1], float(os.environ['ELO']), float(os.environ['EHI']))

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
    lognorm, logNH22, logrel_scat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm = np.transpose(params)
    norm, NH22, rel_scat_norm = 10**lognorm, 10**logNH22, 10**logrel_scat_norm
    Incl = 45 + norm * 0
    Ecut = 400 + norm * 0
    scat_norm = norm * rel_scat_norm

    abs_component = np.einsum('ij,i->ij', 
        absAGN(energies=energies, pars=np.transpose([NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z]), vectorized=True),
        norm)

    scat_component = np.einsum('ij,i->ij',
        tinyxsf.xvec(tinyxsf.x.zpowerlw, energies=energies, pars=np.transpose([PhoIndex, z])),
        scat_norm)

    return abs_component, scat_component

def get_flux(params, e_min, e_max):
    abs_component, scat_component = setup_model(params)
    return tinyxsf.flux.energy_flux(galabso * (abs_component + scat_component), energies, e_min, e_max, axis=1)


# define a likelihood
def loglikelihood(params, plot=False, plot_prefix=''):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm = np.transpose(params)
    abs_component, scat_component = setup_model(params)
    pred_spec = abs_component + scat_component

    pred_counts_src_srcreg = RMF_src.apply_rmf_vectorized(np.einsum('i,i,ji->ji', data['ARF'], galabso, pred_spec))[:,data['chan_mask']] * data['src_expoarea']
    pred_counts_bkg_srcreg = np.einsum('j,i->ij', data['bkg_model_src_region'], bkg_norm) * data['src_expoarea']
    pred_counts_srcreg = pred_counts_src_srcreg + pred_counts_bkg_srcreg
    pred_counts_bkg_bkgreg = np.einsum('j,i->ij', data['bkg_model_bkg_region'], bkg_norm) * data['bkg_expoarea']

    if plot:
        plt.figure()
        l = plt.plot(e_mid, np.einsum('i,ji->ij', e_mid**2 * galabso, pred_spec), ':', color='darkblue', label='src*galabso', mfc='none', alpha=0.3)[0]
        l2 = plt.plot(e_mid, (e_mid**2 * pred_spec).T, '-', label='src', mfc='none', color='k', alpha=0.3)[0]
        l3 = plt.plot(e_mid, (e_mid**2 * abs_component).T, '-', label='torus', mfc='none', color='green', alpha=0.3)[0]
        l4 = plt.plot(e_mid, (e_mid**2 * scat_component).T, '--', label='scat', mfc='none', color='lightblue', alpha=0.3)[0]
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim((e_mid**2 * pred_spec).min() / 5, (e_mid**2 * pred_spec).max() * 2)
        plt.xlabel('Energy [keV]')
        plt.ylabel('Energy Flux Density * keV$^2$ [keV$^2$ * erg/s/cm$^2$/keV]')
        plt.legend([l, l2, l3, l4], ['src*galabso', 'src', 'torus', 'scat'])
        plt.savefig(f'{plot_prefix}src_model_E2.pdf')
        plt.close()

        plt.figure()
        l = plt.plot(e_mid, np.einsum('i,ji->ij', galabso, pred_spec), ':', color='darkblue', label='src*galabso', mfc='none', alpha=0.3)[0]
        l2 = plt.plot(e_mid, pred_spec.T, '-', label='src', mfc='none', color='k', alpha=0.3)[0]
        l3 = plt.plot(e_mid, abs_component.T, '-', label='torus', mfc='none', color='green', alpha=0.3)[0]
        l4 = plt.plot(e_mid, scat_component.T, '--', label='scat', mfc='none', color='lightblue', alpha=0.3)[0]
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(pred_spec.min() / 5, pred_spec.max() * 2)
        plt.xlabel('Energy [keV]')
        plt.ylabel('Energy Flux Density * keV$^2$ [keV$^2$ * erg/s/cm$^2$/keV]')
        plt.legend([l, l2, l3, l4], ['src*galabso', 'src', 'torus', 'scat'])
        plt.savefig(f'{plot_prefix}src_model.pdf')
        plt.close()

        plt.figure()
        plt.plot(data['chan_e_min'], data['src_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='src+bkg')
        plt.plot(data['chan_e_min'], pred_counts_src_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='src')
        plt.plot(data['chan_e_min'], pred_counts_bkg_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig(f'{plot_prefix}src_region_counts.pdf')
        plt.close()

        plt.figure()
        plt.plot(data['chan_e_min'], data['bkg_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_bkg_bkgreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig(f'{plot_prefix}bkg_region_counts.pdf')
        plt.close()

    # compute log Poisson probability
    like_srcreg = tinyxsf.logPoissonPDF_vectorized(pred_counts_srcreg, data['src_region_counts'])
    like_bkgreg = tinyxsf.logPoissonPDF_vectorized(pred_counts_bkg_bkgreg, data['bkg_region_counts'])
    # combined the probabilities. If fitting multiple spectra, you would add them up here as well
    return like_srcreg + like_bkgreg

def main():
    # compute a likelihood:
    z = np.array([data['redshift'], 0, 0.404])
    bkg_norm = np.array([1.0] * 3)
    lognorm = np.log10([1] * 2 + [8.45e-5])
    logscat_norm = np.log10([0.08] * 3)
    PhoIndex = np.array([1.9, 2.0, 2.0])
    TORsigma = np.array([28.0] * 3)
    CTKcover = np.array([0.1] * 3)
    logNH22 = np.log10([0.01] * 3)
    print(loglikelihood(np.transpose([lognorm, logNH22, logscat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm]), plot=True))
    test_fluxes = get_flux(np.transpose([lognorm, logNH22, logscat_norm, PhoIndex, TORsigma, CTKcover, z, bkg_norm]), 0.5, 2)
    print('flux:', test_fluxes)
    print('ARF', np.median(data['ARF']))
    print('src_expoarea:', data['src_expoarea'])
    print('bkg_expoarea:', data['bkg_expoarea'])
    
    ratio_from_models = (data["bkg_model_src_region"].sum() * data["src_expoarea"]) / \
                    (data["bkg_model_bkg_region"].sum() * data["bkg_expoarea"])
    print("bkg source/background count ratio from model:", ratio_from_models)
    print("expected ratio (header-based):", data["src_to_bkg_ratio"])
    assert np.log10(test_fluxes[1] / (2.2211e-09 * u.erg/u.s/u.cm**2)) < 0.2
    assert np.log10(test_fluxes[2] / (9.5175e-14 * u.erg/u.s/u.cm**2)) < 0.2

    # lets define a prior

    PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)
    # we are cool, we can let redshift be a free parameter informed from photo-z
    if 'redshift' in data:
        z_gauss = scipy.stats.norm(data['redshift'], 0.001)
    else:
        z_gauss = scipy.stats.uniform(0, 6)


    # define the prior transform function
    def prior_transform(cube):
        params = cube.copy()
        # uniform from 1e-10 to 1
        params[:,0] = (cube[:,0] * 10 + -10)
        # uniform from 1e-2 to 1e2
        params[:,1] = (cube[:,1] * (4 - -2) + -2)
        params[:,2] = (cube[:,2] * (-1 - -5) + -5)
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
    param_names = ['lognorm', 'logNH22', 'scatnorm', 'PhoIndex', 'TORsigma', 'CTKcover', 'redshift', 'bkg_norm']

    outprefix = sys.argv[1] + '_out_clumpy/'
    # run sampler
    try:
        sampler = ultranest.ReactiveNestedSampler(
            param_names, loglikelihood, prior_transform,
            log_dir=outprefix, resume=True,
            vectorized=True)
    except:
        sampler = ultranest.ReactiveNestedSampler(
            param_names, loglikelihood, prior_transform,
            log_dir=outprefix, resume='overwrite',
            vectorized=True)
    # then to run:
    results = sampler.run(max_num_improvement_loops=0, frac_remain=0.9, Lepsilon=0.1)
    sampler.print_results()
    #sampler.plot()
    samples = results['samples'].copy()
    abs_component, scat_component = setup_model(samples)
    soft_flux_obs = tinyxsf.flux.energy_flux(galabso * (abs_component + scat_component), energies, 0.5, 2, axis=1)
    hard_flux_obs = tinyxsf.flux.energy_flux(galabso * (abs_component + scat_component), energies, 2, 10, axis=1)
    print('soft flux obs:', soft_flux_obs.min(), soft_flux_obs.max(), soft_flux_obs.mean(), soft_flux_obs.std())
    print('hard flux obs:', hard_flux_obs.min(), hard_flux_obs.max(), hard_flux_obs.mean(), hard_flux_obs.std())
    print("setting NH to zero")
    samples[:,1] = -2
    samples[:,2] = -5
    abs_component, scat_component = setup_model(samples)
    soft_flux = tinyxsf.flux.energy_flux(abs_component, energies, 0.5, 2, axis=1)
    hard_flux = tinyxsf.flux.energy_flux(abs_component, energies, 2, 10, axis=1)
    hard_flux_restintr = np.empty(len(samples))
    for i, row in enumerate(samples):
        z = row[6]
        hard_flux_restintr[i] = tinyxsf.flux.energy_flux(abs_component[i, :], energies, 2 / (1 + z), 10 / (1 + z)).to_value(u.erg / u.cm**2 / u.s)

    print('soft flux:', soft_flux.min(), soft_flux.max(), soft_flux.mean(), soft_flux.std())
    print('hard flux:', hard_flux.min(), hard_flux.max(), hard_flux.mean(), hard_flux.std())
    # z, Fint, Fabs, lognorm, Gamma, logNH, fscat
    samples = results['samples'].copy()
    lognorm = samples[:, param_names.index('lognorm')]
    PhoIndex = samples[:, param_names.index('PhoIndex')]
    logNH = samples[:, param_names.index('logNH22')] + 22
    fscat = samples[:, param_names.index('scatnorm')]
    np.savetxt(f'{outprefix}/chains/intrinsic_photon_flux.txt.gz', np.transpose([
        samples[:,6], hard_flux_restintr, soft_flux_obs,
        lognorm, PhoIndex, logNH, fscat,
    ]))

    # and to plot a model:
    loglikelihood(results['samples'][:20,:], plot=True, plot_prefix=f'{sys.argv[1]}_out_clumpy/plots/')

    # and to plot the posterior corner plot:
    sampler.plot()

if __name__ == '__main__':
    main()
