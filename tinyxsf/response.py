"""Functionality for linear instrument response."""
# from https://github.com/dhuppenkothen/clarsach/blob/master/clarsach/respond.py
# GPL licenced code from the Clàrsach project
from functools import partial

import astropy.io.fits as fits
import jax
import numpy as np

__all__ = ["RMF", "ARF", "MockARF"]


def _apply_rmf(spec, in_indices, out_indices, weights, detchans):
    """Apply RMF.

    Parameters
    ----------
    spec: array
        Input spectrum.
    in_indices: array
        List of indices for spec.
    out_indices: array
        list of indices for where to add into output array
    weights: array
        list of weights for multiplying *spec[outindex]* when adding to output array
    detchans: int
        length of output array

    Returns
    -------
    out: array
        Summed entries.
    """
    contribs = spec[in_indices] * weights
    out = jax.numpy.zeros(detchans)
    out = out.at[out_indices].add(contribs)
    return out


def _apply_rmf_vectorized(specs, in_indices, out_indices, weights, detchans):
    """Apply RMF to many spectra.

    Parameters
    ----------
    specs: array
        List of input spectra.
    in_indices: array
        List of indices for spec.
    out_indices: array
        list of indices for where to add into output array
    weights: array
        list of weights for multiplying *spec[outindex]* when adding to output array
    detchans: int
        length of output array

    Returns
    -------
    out: array
        Summed entries. Shape=(len(specs), detchans)
    """
    out = jax.numpy.zeros((len(specs), detchans))

    def body_fun(j, out):
        """Sum up one spectrum.

        Parameters
        ----------
        j: int
            index of spectrum.
        out: array
            will store into out[j,:]

        Returns
        -------
        out_row: array
            returns out[j]
        """
        spec = specs[j]
        contribs = spec[in_indices] * weights
        return out.at[j, out_indices].add(contribs)

    out = jax.lax.fori_loop(0, len(specs), body_fun, out)
    return out


class RMF(object):
    """Response matrix file."""

    def __init__(self, filename):
        """
        Initialise.

        Parameters
        ----------
        filename : str
            The file name with the RMF FITS file
        """
        self._load_rmf(filename)

    def _load_rmf(self, filename):
        """
        Load an RMF from a FITS file.

        Parameters
        ----------
        filename : str
            The file name with the RMF file

        Attributes
        ----------
        n_grp : numpy.ndarray
            the Array with the number of channels in each
            channel set

        f_chan : numpy.ndarray
            The starting channel for each channel group;
            If an element i in n_grp > 1, then the resulting
            row entry in f_chan will be a list of length n_grp[i];
            otherwise it will be a single number

        n_chan : numpy.ndarray
            The number of channels in each channel group. The same
            logic as for f_chan applies

        matrix : numpy.ndarray
            The redistribution matrix as a flattened 1D vector

        energ_lo : numpy.ndarray
            The lower edges of the energy bins

        energ_hi : numpy.ndarray
            The upper edges of the energy bins

        detchans : int
            The number of channels in the detector
        """
        # open the FITS file and extract the MATRIX extension
        # which contains the redistribution matrix and
        # anxillary information
        hdulist = fits.open(filename)

        # get all the extension names
        extnames = np.array([h.name for h in hdulist])

        # figure out the right extension to use
        if "MATRIX" in extnames:
            h = hdulist["MATRIX"]

        elif "SPECRESP MATRIX" in extnames:
            h = hdulist["SPECRESP MATRIX"]
        elif "SPECRESP" in extnames:
            h = hdulist["SPECRESP"]
        else:
            raise AssertionError(f"{extnames} does not contain MATRIX or SPECRESP")

        data = h.data
        hdr = dict(h.header)
        hdulist.close()

        # extract + store the attributes described in the docstring
        n_grp = np.array(data.field("N_GRP"))
        f_chan = np.array(data.field("F_CHAN"))
        n_chan = np.array(data.field("N_CHAN"))
        matrix = np.array(data.field("MATRIX"))

        self.energ_lo = np.array(data.field("ENERG_LO"))
        self.energ_hi = np.array(data.field("ENERG_HI"))
        self.energ_unit = data.columns["ENERG_LO"].unit
        self.detchans = int(hdr["DETCHANS"])
        self.offset = self.__get_tlmin(h)

        # flatten the variable-length arrays
        results = self._flatten_arrays(n_grp, f_chan, n_chan, matrix)
        self.n_grp, self.f_chan, self.n_chan, self.matrix = results
        self.dense_info = None

    def __get_tlmin(self, h):
        """
        Get the tlmin keyword for `F_CHAN`.

        Parameters
        ----------
        h : an astropy.io.fits.hdu.table.BinTableHDU object
            The extension containing the `F_CHAN` column

        Returns
        -------
        tlmin : int
            The tlmin keyword
        """
        # get the header
        hdr = h.header
        # get the keys of all
        keys = np.array(list(hdr.keys()))

        # find the place where the tlmin keyword is defined
        t = np.array(["TLMIN" in k for k in keys])

        # get the index of the TLMIN keyword
        tlmin_idx = np.hstack(np.where(t))[0]

        # get the corresponding value
        tlmin = int(list(hdr.items())[tlmin_idx][1])

        return tlmin

    def _flatten_arrays(self, n_grp, f_chan, n_chan, matrix):
        """Flatten array.

        Parameters
        ----------
        n_grp: array
            number of groups
        f_chan: array
            output start indices
        n_chan: array
            number of output indices
        matrix: array
            weights

        Returns
        -------
        n_grp: array
            number of groups
        f_chan: array
            output start indices
        n_chan: array
            number of output indices
        matrix: array
            weights
        """
        if not len(n_grp) == len(f_chan) == len(n_chan) == len(matrix):
            raise ValueError("Arrays must be of same length!")

        # find all non-zero groups
        nz_idx = n_grp > 0

        # stack all non-zero rows in the matrix
        matrix_flat = np.hstack(matrix[nz_idx], dtype=float)

        # stack all nonzero rows in n_chan and f_chan
        # n_chan_flat = np.hstack(n_chan[nz_idx])
        # f_chan_flat = np.hstack(f_chan[nz_idx])

        # some matrices actually have more elements
        # than groups in `n_grp`, so we'll only pick out
        # those values that have a correspondence in
        # n_grp
        f_chan_new = []
        n_chan_new = []
        for i, t in enumerate(nz_idx):
            if t:
                n = n_grp[i]
                f = f_chan[i]
                nc = n_chan[i]
                if np.size(f) == 1:
                    f_chan_new.append(f.astype(np.int64) - self.offset)
                    n_chan_new.append(nc.astype(np.int64))
                else:
                    f_chan_new.append(f[:n].astype(np.int64) - self.offset)
                    n_chan_new.append(nc[:n].astype(np.int64))

        n_chan_flat = np.hstack(n_chan_new)
        f_chan_flat = np.hstack(f_chan_new)

        return n_grp, f_chan_flat, n_chan_flat, matrix_flat

    def strip(self, channel_mask):
        """Strip response matrix of entries outside the channel mask.

        Parameters
        ----------
        channel_mask : array
            Boolean array indicating which detector channel to keep.

        Returns
        -------
        energy_mask : array
            Boolean array indicating which energy channels were kept.
        """
        n_grp_new = np.zeros_like(self.n_grp)
        n_chan_new = []
        f_chan_new = []
        matrix_new = []
        energ_lo_new = []
        energ_hi_new = []

        in_indices = []
        i_new = 0
        out_indices = []
        weights = []
        k = 0
        resp_idx = 0
        # loop over all channels
        for i in range(len(self.energ_lo)):
            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for current_num_chans, counts_idx in zip(
                self.n_chan[k:k + current_num_groups],
                self.f_chan[k:k + current_num_groups],
            ):
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                # outslice = slice(counts_idx, counts_idx + current_num_chans)
                inslice = slice(resp_idx, resp_idx + current_num_chans)
                mask_valid = channel_mask[counts_idx:counts_idx + current_num_chans]
                if current_num_chans > 0 and mask_valid.any():
                    # add block
                    n_grp_new[i_new] += 1
                    # length
                    n_chan_new.append(current_num_chans)
                    # location in matrix
                    f_chan_new.append(counts_idx)
                    matrix_new.append(self.matrix[inslice])

                    in_indices.append((i_new + np.zeros(current_num_chans, dtype=int))[mask_valid])
                    out_indices.append(np.arange(counts_idx, counts_idx + current_num_chans)[mask_valid])
                    weights.append(self.matrix[inslice][mask_valid])
                resp_idx += current_num_chans

            k += current_num_groups
            if n_grp_new[i_new] > 0:
                energ_lo_new.append(self.energ_lo[i])
                energ_hi_new.append(self.energ_hi[i])
                i_new += 1

        out_indices = np.hstack(out_indices)
        in_indices = np.hstack(in_indices)
        weights = np.hstack(weights)
        self.n_chan = np.array(n_chan_new)
        self.f_chan = np.array(f_chan_new)

        # cut down input array as well
        strip_mask = n_grp_new > 0
        self.n_grp = n_grp_new[strip_mask]
        self.energ_lo = np.array(energ_lo_new)
        self.energ_hi = np.array(energ_hi_new)

        self.matrix = np.hstack(matrix_new)
        i = np.argsort(out_indices)
        # self.dense_info = in_indices[i], out_indices[i], weights[i]
        self.dense_info = in_indices, out_indices, weights
        self._compile()
        return strip_mask

    def _compile(self):
        """Prepare internal functions."""
        if self.dense_info is None:
            return
        in_indices, out_indices, weights = self.dense_info
        self._apply_rmf = jax.jit(
            partial(
                _apply_rmf,
                in_indices=in_indices,
                out_indices=out_indices,
                weights=weights,
                detchans=self.detchans,
            )
        )
        self._apply_rmf_vectorized = jax.jit(
            partial(
                _apply_rmf_vectorized,
                in_indices=in_indices,
                out_indices=out_indices,
                weights=weights,
                detchans=self.detchans,
            )
        )

    def __getstate__(self):
        """Get state for pickling."""
        state = self.__dict__.copy()
        # Remove non-pickleable functions
        if "_apply_rmf" in state:
            del state["_apply_rmf"]
        if "_apply_rmf_vectorized" in state:
            del state["_apply_rmf_vectorized"]
        return state

    def __setstate__(self, state):
        """Restore state from pickling."""
        self.__dict__.update(state)
        self._compile()

    def apply_rmf(self, spec):
        """
        Fold the spectrum through the redistribution matrix.

        The redistribution matrix is saved as a flattened 1-dimensional
        vector to save space. In reality, for each entry in the flux
        vector, there exists one or more sets of channels that this
        flux is redistributed into. The additional arrays `n_grp`,
        `f_chan` and `n_chan` store this information:
            * `n_group` stores the number of channel groups for each
              energy bin
            * `f_chan` stores the *first channel* that each channel
              for each channel set
            * `n_chan` stores the number of channels in each channel
              set

        As a result, for a given energy bin i, we need to look up the
        number of channel sets in `n_grp` for that energy bin. We
        then need to loop over the number of channel sets. For each
        channel set, we look up the first channel into which flux
        will be distributed as well as the number of channels in the
        group. We then need to also loop over the these channels and
        actually use the corresponding elements in the redistribution
        matrix to redistribute the photon flux into channels.

        All of this is basically a big bookkeeping exercise in making
        sure to get the indices right.

        Parameters
        ----------
        spec : numpy.ndarray
            The (model) spectrum to be folded

        Returns
        -------
        counts : numpy.ndarray
            The (model) spectrum after folding, in
            counts/s/channel
        """
        if self.dense_info is not None:
            # in_indices, out_indices, weights = self.dense_info
            # 0.001658s/call; 0.096s/likelihood eval
            # out = np.zeros(self.detchans)
            # np.add.at(out, out_indices, spec[in_indices] * weights)
            # 0.004929s/call; 0.106s/likelihood eval
            # out = np.bincount(out_indices, weights=spec[in_indices] * weights, minlength=self.detchans)
            # 0.001963s/call; 0.077s/likelihood eval
            out = self._apply_rmf(
                spec
            )  # , in_indices, out_indices, weights, self.detchans)
            return out

        # get the number of channels in the data
        nchannels = spec.shape[0]

        # an empty array for the output counts
        counts = np.zeros(nchannels)

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):
            # this is the current bin in the flux spectrum to
            # be folded
            source_bin_i = spec[i]

            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for current_num_chans, counts_idx in zip(
                self.n_chan[k:k + current_num_groups],
                self.f_chan[k:k + current_num_groups]
            ):
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                outslice = slice(counts_idx, counts_idx + current_num_chans)
                inslice = slice(resp_idx, resp_idx + current_num_chans)
                counts[outslice] += self.matrix[inslice] * source_bin_i
                # iterate the response index for next round
                resp_idx += current_num_chans
            k += current_num_groups

        return counts[:self.detchans]

    def apply_rmf_vectorized(self, specs):
        """
        Fold the spectrum through the redistribution matrix.

        The redistribution matrix is saved as a flattened 1-dimensional
        vector to save space. In reality, for each entry in the flux
        vector, there exists one or more sets of channels that this
        flux is redistributed into. The additional arrays `n_grp`,
        `f_chan` and `n_chan` store this information:
            * `n_group` stores the number of channel groups for each
              energy bin
            * `f_chan` stores the *first channel* that each channel
              for each channel set
            * `n_chan` stores the number of channels in each channel
              set

        As a result, for a given energy bin i, we need to look up the
        number of channel sets in `n_grp` for that energy bin. We
        then need to loop over the number of channel sets. For each
        channel set, we look up the first channel into which flux
        will be distributed as well as the number of channels in the
        group. We then need to also loop over the these channels and
        actually use the corresponding elements in the redistribution
        matrix to redistribute the photon flux into channels.

        All of this is basically a big bookkeeping exercise in making
        sure to get the indices right.

        Parameters
        ----------
        specs : numpy.ndarray
            The (model) spectra to be folded

        Returns
        -------
        counts : numpy.ndarray
            The (model) spectrum after folding, in counts/s/channel
        """
        # get the number of channels in the data
        nspecs, nchannels = specs.shape
        if self.dense_info is not None:  # and nspecs < 40:
            in_indices, out_indices, weights = self.dense_info
            out = np.zeros((nspecs, self.detchans))
            for i in range(nspecs):
                # out[i] = np.bincount(out_indices, weights=specs[i,in_indices] * weights, minlength=self.detchans)
                # out[i] = self._apply_rmf(specs[i])
                out[i] = jax.numpy.zeros(self.detchans).at[out_indices].add(specs[i,in_indices] * weights)
            # out = self._apply_rmf_vectorized(specs)
            return out

        # an empty array for the output counts
        counts = np.zeros((nspecs, nchannels))

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):
            # this is the current bin in the flux spectrum to
            # be folded
            source_bin_i = specs[:,i]

            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for current_num_chans, counts_idx in zip(
                self.n_chan[k:k + current_num_groups],
                self.f_chan[k:k + current_num_groups]
            ):
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                to_add = np.outer(
                    source_bin_i, self.matrix[resp_idx:resp_idx + current_num_chans]
                )
                counts[:, counts_idx:counts_idx + current_num_chans] += to_add

                # iterate the response index for next round
                resp_idx += current_num_chans
            k += current_num_groups

        return counts[:, : self.detchans]

    def get_dense_matrix(self):
        """Extract the redistribution matrix as a dense numpy matrix.

        The redistribution matrix is saved as a 1-dimensional
        vector to save space (see apply_rmf for more information).
        This function converts it into a dense array.

        Returns
        -------
        dense_matrix : numpy.ndarray
            The RMF as a dense 2d matrix.
        """
        # get the number of channels in the data
        nchannels = len(self.energ_lo)
        nenergies = self.detchans

        # an empty array for the output counts
        dense_matrix = np.zeros((nchannels, nenergies))

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):
            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for _ in range(current_num_groups):
                current_num_chans = int(self.n_chan[k])
                # get the right index for the start of the counts array
                # to put the data into
                counts_idx = self.f_chan[k]
                # this is the current number of channels to use

                k += 1

                # assign the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels

                outslice = slice(counts_idx, counts_idx + current_num_chans)
                inslice = slice(resp_idx, resp_idx + current_num_chans)
                dense_matrix[i, outslice] = self.matrix[inslice]

                # iterate the response index for next round
                resp_idx += current_num_chans

        return dense_matrix


class ARF(object):
    """Area response file."""

    def __init__(self, filename):
        """Initialise.

        Parameters
        ----------
        filename : str
            The file name with the ARF file
        """
        self._load_arf(filename)
        pass

    def _load_arf(self, filename):
        """Load an ARF from a FITS file.

        Parameters
        ----------
        filename : str
            The file name with the ARF file
        """
        # open the FITS file and extract the MATRIX extension
        # which contains the redistribution matrix and
        # anxillary information
        hdulist = fits.open(filename)
        h = hdulist["SPECRESP"]
        data = h.data
        hdr = h.header
        hdulist.close()

        # extract + store the attributes described in the docstring

        self.e_low = np.array(data.field("ENERG_LO"))
        self.e_high = np.array(data.field("ENERG_HI"))
        self.e_unit = data.columns["ENERG_LO"].unit
        self.specresp = np.array(data.field("SPECRESP"))

        if "EXPOSURE" in list(hdr.keys()):
            self.exposure = float(hdr["EXPOSURE"])
        else:
            self.exposure = 1.0

        if "FRACEXPO" in data.columns.names:
            self.fracexpo = float(data["FRACEXPO"])
        else:
            self.fracexpo = 1.0

    def strip(self, mask):
        """Remove unneeded energy ranges.

        Parameters
        ----------
        mask: array
            Boolean array indicating which energy channel to keep.
        """
        self.e_low = self.e_low[mask]
        self.e_high = self.e_high[mask]
        self.specresp = self.specresp[mask]

    def apply_arf(self, spec, exposure=None):
        """Fold the spectrum through the ARF.

        The ARF is a single vector encoding the effective area information
        about the detector. A such, applying the ARF is a simple
        multiplication with the input spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            The (model) spectrum to be folded

        exposure : float, default None
            Value for the exposure time. By default, `apply_arf` will use the
            exposure keyword from the ARF file. If this exposure time is not
            correct (for example when simulated spectra use a different exposure
            time and the ARF from a real observation), one can override the
            default exposure by setting the `exposure` keyword to the correct
            value.

        Returns
        -------
        s_arf : numpy.ndarray
            The (model) spectrum after folding, in
            counts/s/channel
        """
        assert spec.shape[0] == self.specresp.shape[0], (
            "Input spectrum and ARF must be of same size.",
            spec.shape,
            self.specresp.shape,
        )
        e = self.exposure if exposure is None else exposure
        return spec * self.specresp * e


class MockARF(ARF):
    """Mock area response file."""

    def __init__(self, rmf):
        """Initialise.

        Parameters
        ----------
        rmf: RMF
            RMF object to mimic.
        """
        self.e_low = rmf.energ_lo
        self.e_high = rmf.energ_hi
        self.e_unit = rmf.energ_unit
        self.exposure = None
        self.specresp = np.ones_like(self.e_low)
