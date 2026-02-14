import sys
import os
from astropy.cosmology import LambdaCDM
from astropy import units as u
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from optns.profilelike import ComponentModel
import tinyxsf
from tinyxsf.flux import luminosity
from tinyxsf.model import FixedTable

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.730)


# tinyxsf.x.chatter(0)
tinyxsf.x.abundance('wilm')
tinyxsf.x.cross_section('vern')

PhoIndex_grid = np.arange(1, 3.1, 0.1)
PhoIndex_grid[-1] = 3.0
logNH_grid = np.arange(20, 25.1, 0.1)

filename = sys.argv[1]

elo = 0.3
ehi = 8
data = tinyxsf.load_pha(filename, elo, ehi)
# fetch some basic information about our spectrum
e_lo = data['e_lo']
e_hi = data['e_hi']
#e_mid = (data['e_hi'] + data['e_lo']) / 2.
#e_width = data['e_hi'] - data['e_lo']
energies = np.append(e_lo, e_hi[-1])
RMF_src = data['RMF_src']
#chan_e = (data['chan_e_min'] + data['chan_e_max']) / 2.

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
galabso = tinyxsf.x.TBabs(energies=energies, pars=[data['galnh']])

z = data['redshift']
absAGN = FixedTable(
    os.path.join(os.environ.get('MODELDIR', '.'), 'uxclumpy-cutoff.fits'),
    energies=energies, redshift=z)

Nsrc_chan = len(data['src_region_counts'])
Nbkg_chan = len(data['bkg_region_counts'])
counts_flat = np.hstack((data['src_region_counts'], data['bkg_region_counts']))
nonlinear_param_names = ['logNH', 'PhoIndex']
linear_param_names = ['norm_src', 'norm_bkg']
# create OptNS object, and give it all of these ingredients,
# as well as our data
statmodel = ComponentModel(2, counts_flat)

TORsigma = 28.0
CTKcover = 0.1
Incl = 45.0
Ecut = 400

def compute_model_components(params):
    logNH, PhoIndex = params
    # first component: a absorbed power law
    plabso = absAGN(energies=energies, pars=[10**(logNH - 22), PhoIndex, Ecut, TORsigma, CTKcover, Incl])

    # now we need to project all of our components through the response.
    src_components = data['ARF'] * galabso * plabso
    pred_counts_src_srcreg = RMF_src.apply_rmf(src_components)[data['chan_mask']] * data['src_expoarea']
    # add non-folded background to source region components
    pred_counts = np.zeros((2, Nsrc_chan + Nbkg_chan))
    # the three folded source components in the source region
    pred_counts[0, :Nsrc_chan] = pred_counts_src_srcreg
    # the unfolded background components in the source region
    pred_counts[1, :Nsrc_chan] = data['bkg_model_src_region'] * data['src_expoarea']
    # the unfolded background components in the background region
    pred_counts[1, Nsrc_chan:] = data['bkg_model_bkg_region'] * data['bkg_expoarea']
    # notice how the source does not affect the background:
    #   pred_counts[0, Nsrc_chan:] = 0  # they remain zero
    return pred_counts.T


profile_like = np.zeros((len(PhoIndex_grid), len(logNH_grid)))
Lint = np.zeros((len(PhoIndex_grid), len(logNH_grid)))
extent = [logNH_grid.min(), logNH_grid.max(), PhoIndex_grid.min(), PhoIndex_grid.max()]

for i, PhoIndex in enumerate(tqdm.tqdm(PhoIndex_grid)):
    for j, logNH in enumerate(logNH_grid):
        X = compute_model_components([logNH, PhoIndex])
        res = statmodel.loglike_poisson_optimize(X)
        norms = res.x
        profile_like[i,j] = -res.fun
        srcnorm = np.exp(norms[0])
        Lint[i,j] = np.log10(luminosity(
            srcnorm * tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, z]),
            energies, 2, 10, z, cosmo
        ) / (u.erg/u.s))
# plot likelihoods and 
plt.imshow(Lint, vmin=42, vmax=47, extent=extent, cmap='rainbow')
plt.colorbar(orientation='horizontal')
plt.savefig(f'{filename}_L.pdf')
plt.close()

plt.imshow(-2 * (profile_like - profile_like.max()), vmin=0, vmax=10, extent=extent, origin='upper', cmap='Greys_r')
plt.colorbar(orientation='horizontal')
plt.contour(-2 * (profile_like - profile_like.max()), levels=[1, 2, 3], extent=extent, origin='upper', colors=['k'] * 3)
plt.savefig(f'{filename}_like.pdf')
plt.close()

# compute posterior distribution of logNH, L
prob = np.exp(profile_like - profile_like.max())
logNH_probs = np.mean(prob, axis=0)
logNH_probs /= logNH_probs.sum()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7, 2))
axs[0].plot(10**logNH_grid, logNH_probs / logNH_probs.max())
Lgrid = np.arange(max(42, Lint.min() - 0.1), Lint.max() + 0.1, 0.05)
Lgrid_probs = 0 * Lgrid
for i, PhoIndex in enumerate(tqdm.tqdm(PhoIndex_grid)):
    for j, logNH in enumerate(logNH_grid):
        k = np.argmin(np.abs(Lint[i,j] - Lgrid))
        Lgrid_probs[k] += prob[i,j]
axs[1].plot(10**Lgrid, Lgrid_probs / Lgrid_probs.max())
axs[0].set_xlabel(r'Column density $N_\mathrm{H}$ [#/cm$^2$]')
axs[1].set_xlabel(r'Luminosity (2-10keV, intr.) [erg/s]')
axs[0].set_xscale('log')
axs[1].set_xscale('log')
#axs[0].set_xticks(10**np.arange(20, 26))
#axs[1].set_xticks(10**np.arange(int(Lgrid.min()), int(Lgrid.max())))
axs[0].set_yticks([0,1])
plt.savefig(f'{filename}_prob.pdf')
plt.savefig(f'{filename}_prob.png')
plt.close()

np.savetxt(f'{filename}_like.txt', profile_like, fmt='%.3f')
np.savetxt(f'{filename}_L.txt', [Lgrid, Lgrid_probs], fmt='%.6f')
np.savetxt(f'{filename}_NH.txt', [logNH_grid, logNH_probs], fmt='%.6f')
