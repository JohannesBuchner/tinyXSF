"""Statistical and astrophysical models."""
import hashlib
import itertools

import astropy.io.fits as pyfits
import numpy as np
import tqdm
import xspec_models_cxc as x
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gammaln

from joblib import Memory

mem = Memory('.', verbose=False)


def logPoissonPDF_vectorized(models, counts):
    """Compute poisson probability.

    Parameters
    ----------
    models: array
        expected number of counts. shape is (num_models, len(counts)).
    counts: array
        observed counts (non-negative integer)

    Returns
    -------
    loglikelihood: array
        ln of the Poisson likelihood, neglecting the factorial(counts) factor,
        shape=(num_models,).
    """
    log_models = np.log(np.clip(models, 1e-100, None))
    return np.sum(log_models * counts.reshape((1, -1)), axis=1) - models.sum(axis=1)


def logPoissonPDF(model, counts):
    """Compute poisson probability.

    Parameters
    ----------
    model: array
        expected number of counts
    counts: array
        observed counts (non-negative integer)

    Returns
    -------
    loglikelihood: float
        ln of the Poisson likelihood, neglecting the factorial(counts) factor.
    """
    log_model = np.log(np.clip(model, 1e-100, None))
    return np.sum(log_model * counts) - model.sum()


def logNegBinomialPDF(model, counts, k):
    """Compute Negative Binomial probability (Poisson-Gamma mixture).

    Parameters
    ----------
    model: array
        expected number of counts (mean mu)
    counts: array
        observed counts (non-negative integer)
    k: float
        dispersion parameter k > 0 (larger -> closer to Poisson)

    Returns
    -------
    loglikelihood: float
        ln of the Negative Binomial likelihood, neglecting constant factors
        that depend only on the data (counts).
    """
    mu = np.asarray(model, float)
    y = np.asarray(counts, float)

    mu = np.clip(mu, 1e-100, None)

    # log p(y | mu, k) up to constants in y:
    # ln Γ(y+k) - ln Γ(k)  + k ln(k/(k+mu)) + y ln(mu/(k+mu))
    return float(np.sum(gammaln(y + k) - gammaln(k) + k * (np.log(k) - np.log(k + mu)) + y * (np.log(mu) - np.log(k + mu))))


def xvec(model, energies, pars):
    """Evaluate a model in a vectorized way.

    Parameters
    ----------
    model: object
        xspec model (from tinyxsf.x module, which is xspec_models_cxc)
    energies: array
        energies in keV where to evaluate model
    pars: array
        list of parameter vectors

    Returns
    -------
    results: array
        for each parameter vector in pars, evaluates the model
        at the given energies. Has shape (pars.shape[0], energies.shape[0])
    """
    results = np.empty((len(pars), len(energies) - 1))
    for i, pars_i in enumerate(pars):
        model(energies=energies, pars=pars_i, out=results[i, :])
    return results


@mem.cache
def check_if_sorted(param_vals, parameter_grid):
    """Check if parameters are stored in a sorted way.

    Parameters
    ----------
    param_vals: array
        list of parameter values stored
    parameter_grid: array
        list of possible values for each parameter

    Returns
    -------
    sorted: bool
        True if param_vals==itertools.product(*parameter_grid)
    """
    for i, params in enumerate(itertools.product(*parameter_grid)):
        if not np.all(param_vals[i] == params):
            return False
    return True


def hashfile(filename):
    """Compute a hash for the content of a file.

    Parameters
    ----------
    filename: str
        file name

    Returns
    -------
    hash: str
        hash digest of file content
    """
    with open(filename, 'rb', buffering=0) as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@mem.cache
def _load_table(filename, filehash=None):
    """Load data from table file.

    Parameters
    ----------
    filename: str
        filename of a OGIP FITS file.
    filehash: str
        hash of the file

    Returns
    -------
    parameter_grid: list
        list of values for each parameter
    data: array
        array of shape `(len(g) for g in parameter_grid)`
        containing the spectra for each parameter grid point.
    info: dict
        information about the table, including parameter_names, name,
        e_model_lo, e_model_hi, e_model_mid, deltae, parameter_grid
    """
    f = pyfits.open(filename)
    assert f[0].header["MODLUNIT"] in ("photons/cm^2/s", "ergs/cm**2/s")
    assert f[0].header["HDUCLASS"] == "OGIP"
    self = {}
    self['parameter_names'] = f["PARAMETERS"].data["NAME"]
    self['name'] = f[0].header["MODLNAME"]
    parameter_grid = [
        row["VALUE"][: row["NUMBVALS"]] for row in f["PARAMETERS"].data
    ]
    self['e_model_lo'] = f["ENERGIES"].data["ENERG_LO"]
    self['e_model_hi'] = f["ENERGIES"].data["ENERG_HI"]
    self['e_model_mid'] = (self['e_model_lo'] + self['e_model_hi']) / 2.0
    self['deltae'] = self['e_model_hi'] - self['e_model_lo']
    specdata = f["SPECTRA"].data
    is_sorted = check_if_sorted(specdata["PARAMVAL"], parameter_grid)
    shape = tuple([len(g) for g in parameter_grid] + [len(specdata["INTPSPEC"][0])])

    if is_sorted:
        data = specdata["INTPSPEC"].reshape(shape)
    else:
        data = np.nan * np.zeros(
            [len(g) for g in parameter_grid] + [len(specdata["INTPSPEC"][0])]
        )
        for index, params, row in zip(
            tqdm.tqdm(
                list(itertools.product(*[range(len(g)) for g in parameter_grid]))
            ),
            itertools.product(*parameter_grid),
            sorted(specdata, key=lambda row: tuple(row["PARAMVAL"])),
        ):
            np.testing.assert_allclose(params, row["PARAMVAL"])
            data[index] = row["INTPSPEC"]
    assert np.isfinite(data).all(), data
    return parameter_grid, data, self


class Table:
    """Additive or multiplicative table model."""

    def __init__(self, filename, method="linear", verbose=True):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        method: str
            interpolation kind, passed to RegularGridInterpolator
        verbose: bool
            whether to print information about the table
        """
        parameter_grid, data, info = _load_table(filename, hashfile(filename))
        self.__dict__.update(info)
        if verbose:
            print(f'ATABLE "{self.name}"')
            for param_name, param_values in zip(self.parameter_names, parameter_grid):
                print(f"    {param_name}: {param_values.tolist()}")
        self.interpolator = RegularGridInterpolator(parameter_grid, data, method=method)

    def __call__(self, energies, pars, vectorized=False):
        """Evaluate spectrum.

        Parameters
        ----------
        energies: array
            energies in keV where spectrum should be computed
        pars: list
            parameter values.
        vectorized: bool
            if true, pars is a list of parameter vectors,
            and the function returns a list of spectra.

        Returns
        -------
        spectrum: array
            photon spectrum, corresponding to the parameter values,
            one entry for each value in energies in phot/cm^2/s.
        """
        if vectorized:
            z = pars[:, -1]
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
            delta_e = e_hi - e_lo
            try:
                model_int_spectrum = self.interpolator(pars[:, :-1])
            except ValueError as e:
                for pname, gridpoints, values in zip(self.parameter_names, self.interpolator.grid, pars[:, :-1].transpose()):
                    lo = gridpoints.min()
                    hi = gridpoints.max()
                    errstr = f"Error for table model '{self.name}: requested evaluation for parameter '{pname}' at value "
                    if not np.all(values >= lo):
                        raise ValueError(f"{errstr}{values.min()}, but lowest tabulated value is {lo}") from e
                    if not np.all(values <= hi):
                        raise ValueError(f"{errstr}{values.max()}, but highest tabulated value is {lo}") from e
                raise e

            results = np.empty((len(z), len(e_mid)))
            for i, zi in enumerate(z):
                # this model spectrum contains for each bin [e_lo...e_hi] the integral of energy
                # now we have a new energy, energies
                results[i, :] = (
                    np.interp(
                        # look up in rest-frame, which is at higher energies at higher redshifts
                        x=e_mid * (1 + zi),
                        # in the model spectral grid
                        xp=self.e_model_mid,
                        # use spectral density, which is stretched out if redshifted.
                        fp=model_int_spectrum[i, :] / self.deltae * (1 + zi),
                    ) * delta_e / (1 + zi)
                )
            return results
        else:
            z = pars[-1]
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
            delta_e = e_hi - e_lo
            (model_int_spectrum,) = self.interpolator([pars[:-1]])
            # this model spectrum contains for each bin [e_lo...e_hi] the integral of energy
            # now we have a new energy, energies
            return (
                np.interp(
                    # look up in rest-frame, which is at higher energies at higher redshifts
                    x=e_mid * (1 + z),
                    # in the model spectral grid
                    xp=self.e_model_mid,
                    # use spectral density, which is stretched out if redshifted.
                    fp=model_int_spectrum / self.deltae * (1 + z),
                ) * delta_e / (1 + z)
            )


@mem.cache
def _load_redshift_interpolated_table(filename, filehash, energies, redshift, fix={}):
    """Load data from table file, precomputed for a energy grid.

    Parameters
    ----------
    filename: str
        filename of a OGIP FITS file.
    filehash: str
        hash of the file
    energies: array
        Energies at which to compute model.
    redshift: float
        redshift to use for precomputing
    fix: dict
        dictionary of parameter names and their values to fix
        for faster data loading.

    Returns
    -------
    newshape: tuple
        multidimensional shape of data according to info['parameter_grid']
    data: array
        data table
    info: dict
        information about the table, including parameter_names, name, e_model_lo, e_model_hi, e_model_mid, deltae, parameter_grid
    """
    parameter_grid, data, info = _load_table(filename, filehash)
    # interpolate data from original energy grid onto new energy grid
    e_lo = energies[:-1]
    e_hi = energies[1:]
    e_mid = (e_lo + e_hi) / 2.0
    delta_e = e_hi - e_lo
    # look up in rest-frame, which is at higher energies at higher redshifts
    e_mid_rest = e_mid * (1 + redshift)
    deltae_rest = delta_e / (1 + redshift)
    e_model_mid = info['e_model_mid']
    info['energies'] = energies
    model_deltae_rest = info['deltae'] / (1 + redshift)

    # param_shapes = [len(p) for p in parameter_grid]
    # ndim = len(param_shapes)

    # Flatten the parameter grid into indices
    data_reshaped = data.reshape((-1, data.shape[-1]))
    n_points = data_reshaped.shape[0]
    mask = np.ones(n_points, dtype=bool)

    # Precompute grids
    param_grids = np.meshgrid(*parameter_grid, indexing='ij')
    # same shape as data without last dim
    param_grids_flat = [g.flatten() for g in param_grids]
    # each entry is flattened to match reshaped data
    parameter_names = [str(pname) for pname in info['parameter_names']]

    # Now apply fix conditions
    for pname, val in fix.items():
        assert pname in parameter_names, (pname, parameter_names)
        param_idx = parameter_names.index(pname)
        mask &= (param_grids_flat[param_idx] == val)
        assert mask.any(), (f'You can only fix parameter {pname} to one of:', parameter_grid[param_idx])

    # Mask valid rows
    valid_data = data_reshaped[mask]
    # Build new parameter grid (only for unfixed parameters)
    newparameter_grid = []
    for p, pname in zip(parameter_grid, info['parameter_names']):
        if pname not in fix:
            newparameter_grid.append(p)

    # Interpolate
    newdata = np.zeros((valid_data.shape[0], len(e_mid_rest)))
    for i, row in enumerate(valid_data):
        newdata[i, :] = np.interp(
            x=e_mid_rest,
            xp=e_model_mid,
            fp=row / model_deltae_rest,
        ) * deltae_rest

    info['parameter_grid'] = newparameter_grid
    newshape = tuple([len(g) for g in newparameter_grid] + [len(e_mid_rest)])
    return newshape, newdata, info


@mem.cache
def _load_redshift_interpolated_table_folded(filename, filehash, energies, redshift, ARF, RMF, fix={}):
    """Load data from table file, and fold it through the response.

    Parameters
    ----------
    filename: str
        filename of a OGIP FITS file.
    filehash: str
        hash of the file
    energies: array
        Energies at which to compute model.
    redshift: float
        redshift to use for precomputing
    ARF: ARF
        area response function
    RMF: RMF
        response matrix
    fix: dict
        dictionary of parameter names and their values to fix
        for faster data loading.

    Returns
    -------
    newshape: tuple
        multidimensional shape of data according to info['parameter_grid']
    data: array
        data table
    info: dict
        information about the table, including parameter_names, name, e_model_lo, e_model_hi, e_model_mid, deltae, parameter_grid
    """
    oldshape, olddata, info = _load_redshift_interpolated_table(
        filename, filehash, energies, redshift=redshift, fix=fix)
    newshape = list(oldshape)
    newshape[-1] = RMF.detchans
    newdata = RMF.apply_rmf_vectorized(olddata * ARF)
    assert newdata.shape == (len(olddata), RMF.detchans), (newdata.shape, olddata.shape, len(olddata), newshape[-1])
    return newshape, newdata.reshape(newshape), info


class FixedTable(Table):
    """Additive or multiplicative table model with fixed energy grid."""

    def __init__(self, filename, energies, redshift=0, method="linear", fix={}, verbose=True):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        energies: array
            energies in keV where spectrum should be computed
        redshift: float
            Redshift
        method: str
            interpolation kind, passed to RegularGridInterpolator
        fix: dict
            dictionary of parameter names and their values to fix
            for faster data loading.
        verbose: bool
            whether top print information about the table
        """
        shape, data, info = _load_redshift_interpolated_table(
            filename, hashfile(filename), energies, redshift=redshift, fix=fix)
        self.__dict__.update(info)
        if verbose:
            print(f'ATABLE "{self.name}" (redshift={redshift})')
            for param_name, param_values in zip(self.parameter_names, self.parameter_grid):
                print(f"    {param_name}: {param_values.tolist()}")
        self.interpolator = RegularGridInterpolator(
            self.parameter_grid, data.reshape(shape),
            method=method)

    def __call__(self, pars, vectorized=False, energies=None):
        """Evaluate spectrum.

        Parameters
        ----------
        pars: list
            parameter values.
        vectorized: bool
            if true, pars is a list of parameter vectors,
            and the function returns a list of spectra.
        energies: array
            energies in keV where spectrum should be computed (ignored)

        Returns
        -------
        spectrum: array
            photon spectrum, corresponding to the parameter values,
            one entry for each value in energies in phot/cm^2/s.
        """
        assert np.size(vectorized) == 1
        assert np.ndim(vectorized) == 0
        if energies is not None:
            assert np.all(self.energies == energies)
        if vectorized:
            assert np.ndim(pars) == 2
            try:
                return self.interpolator(pars)
            except ValueError as e:
                raise ValueError(f"Interpolator with parameters {self.parameter_names} called with {pars}") from e
        else:
            assert np.ndim(pars) == 1
            try:
                return self.interpolator([pars])[0]
            except ValueError as e:
                pars_assigned = ' '.join([f'{k}={v}' for k, v in zip(self.parameter_names, pars)])
                raise ValueError(f"Interpolator called with {pars_assigned}") from e


class FixedFoldedTable(FixedTable):
    """Additive or multiplicative table model folded through response."""

    def __init__(self, filename, energies, RMF, ARF, redshift=0, method="linear", fix={}, verbose=True):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        energies: array
            energies in keV where spectrum should be computed
        redshift: float
            Redshift
        method: str
            interpolation kind, passed to RegularGridInterpolator
        fix: dict
            dictionary of parameter names and their values to fix
            for faster data loading.
        RMF: RMF
            response matrix
        ARF: ARF
            area response function
        verbose: bool
            whether to print information about the table
        """
        shape, data, info = _load_redshift_interpolated_table_folded(
            filename, hashfile(filename), energies, redshift=redshift, fix=fix,
            RMF=RMF, ARF=ARF)
        self.__dict__.update(info)
        if verbose:
            print(f'ATABLE "{self.name}" (redshift={redshift})')
            for param_name, param_values in zip(self.parameter_names, self.parameter_grid):
                print(f"    {param_name}: {param_values.tolist()}")
        self.interpolator = RegularGridInterpolator(
            self.parameter_grid, data.reshape(shape),
            method=method)


def prepare_folded_model0d(model, energies, pars, ARF, RMF, nonnegative=True):
    """Prepare a folded spectrum.

    Parameters
    ----------
    model: object
        xspec model (from tinyxsf.x module, which is xspec_models_cxc)
    energies: array
        energies in keV where spectrum should be computed (ignored)
    pars: list
        parameter values.
    ARF: array
        vector for multiplication before applying the RMF
    RMF: RMF
        RMF object for folding
    nonnegative: bool
        <MEANING OF nonnegative>

    Returns
    -------
    folded_spectrum: array
        folded spectrum after applying RMF & ARF
    """
    if nonnegative:
        return RMF.apply_rmf(np.clip(model(energies=energies, pars=pars), 0, None) * ARF)
    else:
        return RMF.apply_rmf(model(energies=energies, pars=pars) * ARF)


@mem.cache(ignore=['model'])
def _prepare_folded_model1d(model, modelname, energies, pars, ARF, RMF, nonnegative=True):
    """Prepare a function that returns the folded model.

    Parameters
    ----------
    model: object
        xspec model (from tinyxsf.x module, which is xspec_models_cxc)
    modelname: str
        name of xspec model
    energies: array
        energies in keV where spectrum should be computed (ignored)
    pars: list
        parameter values; one of the entries can be an array,
        which will be the interpolation range.
    ARF: array
        vector for multiplication before applying the RMF
    RMF: RMF
        RMF object for folding
    nonnegative: bool
        <MEANING OF nonnegative>

    Returns
    -------
    freeparam_grid: tuple
        the pars element which is an array.
    folded_spectrum: array
        folded spectrum after applying RMF & ARF
    """
    mask_fixed = np.array([np.size(p) == 1 for p in pars])
    assert (~mask_fixed).sum() == 1, mask_fixed
    i_variable = np.where(~(mask_fixed))[0][0]
    assert pars[i_variable].ndim == 1

    data = np.zeros((len(pars[i_variable]), len(ARF)))
    for i, variable in enumerate(pars[i_variable]):
        pars_row = list(pars)
        pars_row[i_variable] = variable
        data[i] = model(energies=energies, pars=pars_row)
    foldeddata = RMF.apply_rmf_vectorized(data * ARF)
    if nonnegative:
        foldeddata = np.clip(foldeddata, 0, None)
    return pars[i_variable], foldeddata


def prepare_folded_model1d(model, energies, pars, ARF, RMF, nonnegative=True, method='linear'):
    """Prepare a function that returns the folded model.

    Parameters
    ----------
    model: object
        xspec model (from tinyxsf.x module, which is xspec_models_cxc)
    energies: array
        energies in keV where spectrum should be computed (ignored)
    pars: list
        parameter values; one of the entries can be an array,
        which will be the interpolation range.
    ARF: array
        vector for multiplication before applying the RMF
    RMF: RMF
        RMF object for folding
    nonnegative: bool
        <MEANING OF nonnegative>
    method: str
        interpolation kind, passed to RegularGridInterpolator

    Returns
    -------
    simple_interpolator: func
        function that given the free parameter value returns a folded spectrum.
    """
    grid, foldeddata = _prepare_folded_model1d(
        model=model, modelname=model.__name__, pars=pars,
        energies=energies, ARF=ARF, RMF=RMF, nonnegative=True)
    interp = RegularGridInterpolator((grid,), foldeddata, method=method)

    def simple_interpolator(par):
        """Interpolator for a single parameter.

        Parameters
        ----------
        par: float
            The value for the one free model parameter

        Returns
        -------
        spectrum: array
            photon spectrum
        """
        try:
            return interp([par])[0]
        except ValueError as e:
            raise ValueError(f'invalid parameter value passed: {par}') from e

    return simple_interpolator
