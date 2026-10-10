"""Functionality for OGIP grouping of spectra."""
import numpy as np


def read_grouping(header, data):
    """Read the OGIP GROUPING and QUALITY columns of a SPECTRUM extension.

    Parameters
    ----------
    header: astropy header
        Header of the SPECTRUM extension.
    data: astropy table
        Data of the SPECTRUM extension.

    Returns
    -------
    grouping, quality: arrays or None
        Column values, one entry per detector channel; None if the
        column is absent. Per OGIP, a GROUPING keyword value of 0 means
        that no grouping is defined and the column is ignored.
    """
    grouping = None
    if "GROUPING" in data.columns.names and header.get("GROUPING") != 0:
        grouping = np.asarray(data["GROUPING"])
    quality = None
    if "QUALITY" in data.columns.names:
        quality = np.asarray(data["QUALITY"])
    return grouping, quality


def parse_grouping(grouping):
    """Parse an OGIP GROUPING column into groups of channel indices.

    The column marks each detector channel with 1 (+1) if it is the
    first channel of a group, -1 if it is a subsequent channel of the
    same group, and 0 if it is not used. Groups are contiguous channel
    ranges.

    Some tools mark the *last* channel of each group with 1 instead and
    the other members with -1. This convention is detected when the
    standard interpretation fails (a -1 appears without an open group)
    and handled accordingly; the standard interpretation takes
    precedence when both are possible.

    Parameters
    ----------
    grouping: array
        GROUPING column values, one entry per detector channel.

    Returns
    -------
    groups: list of arrays
        Channel indices (0-based row positions) of each group,
        in increasing order.
    """
    grouping = np.asarray(grouping)
    for closing_value in (None, 1):
        # standard OGIP (closing_value None): 1 opens a group, -1 continues it
        # alternative: -1 opens/continues a group, 1 closes it
        # (the closing channel belongs to the group)
        groups = []
        current = None
        failed = False
        for i, value in enumerate(grouping):
            if value == 0:
                # channel not used, closes an open group
                current = None
            elif value == 1:
                if closing_value is not None and current is not None:
                    # channel is a further member and closes the group
                    current.append(i)
                    current = None
                else:
                    # start a new group
                    current = [i]
                    groups.append(current)
                    if closing_value is not None:
                        # single-channel group, already closed
                        current = None
            elif value == -1:
                if current is None:
                    if closing_value is None:
                        # continuation of a group that was never opened
                        failed = True
                        break
                    # alternative convention: first member of a new group
                    current = [i]
                    groups.append(current)
                else:
                    current.append(i)
            else:
                raise ValueError(
                    "invalid GROUPING value %r at channel %d" % (value, i))
        if not failed:
            return [np.array(g, dtype=int) for g in groups]
    raise ValueError("invalid GROUPING column: %s" % (grouping,))


def get_bins(grouping, quality, mask, nchan):
    """Compute the analysis bins from the OGIP grouping information.

    The bins are the groups of the GROUPING column (one bin per channel
    if there is no grouping), restricted to the channel mask and to
    good-quality channels. Groups reaching past the mask contribute
    only the channels inside it.

    Parameters
    ----------
    grouping: array or None
        GROUPING column values.
    quality: array or None
        QUALITY column values; channels with non-zero quality are excluded.
    mask: array
        Boolean array indicating which detector channels to use
        (e.g. the requested energy range).
    nchan: int
        Number of detector channels.

    Returns
    -------
    bins: list of arrays or None
        Channel indices (0-based) of each bin, in increasing order, or
        None if there is no grouping and no quality information.
    """
    if grouping is None and quality is None:
        return None
    if grouping is None:
        raw_groups = [np.array([i]) for i in range(nchan)]
    else:
        raw_groups = parse_grouping(grouping)
    bins = []
    for group in raw_groups:
        keep = [c for c in group if mask[c] and (quality is None or quality[c] == 0)]
        if len(keep) > 0:
            bins.append(np.array(keep, dtype=int))
    if len(bins) == 0:
        raise ValueError("no good channels left in the requested energy range")
    return bins


def group_of_channel(bins, nchan):
    """Map each detector channel to its bin index, or -1 if it is not used.

    Parameters
    ----------
    bins: list of arrays
        Channel indices of each bin, in increasing order.
    nchan: int
        Number of detector channels.

    Returns
    -------
    group_of_channel: array of int
        Bin index per detector channel, -1 for unused channels.
    """
    mapping = np.full(nchan, -1, dtype=int)
    for i, group in enumerate(bins):
        mapping[group] = i
    return mapping


def sum_bins(values, bins, dtype=None):
    """Sum a per-channel array over the channels of each bin.

    Parameters
    ----------
    values: array
        Per-channel values (counts or spectra).
    bins: list of arrays
        Channel indices of each bin.
    dtype: dtype or None
        Output dtype, e.g. int for counts.

    Returns
    -------
    summed: array
        Sum over the channels of each bin.
    """
    return np.array([values[group].sum() for group in bins], dtype=dtype)


def bin_edges(chan_e_min, chan_e_max, bins):
    """Energy range of each bin: the outermost of its channels.

    Parameters
    ----------
    chan_e_min, chan_e_max: arrays
        Per-channel energy edges.
    bins: list of arrays
        Channel indices of each bin.

    Returns
    -------
    e_min, e_max: arrays
        Energy edges of each bin.
    """
    e_min = np.array([chan_e_min[group].min() for group in bins])
    e_max = np.array([chan_e_max[group].max() for group in bins])
    return e_min, e_max


def group_matrix(m, bins):
    """Combine the columns of a dense response matrix per bin.

    Parameters
    ----------
    m: array
        Dense response matrix (energy bins x detector channels).
    bins: list of arrays
        Channel indices of each bin.

    Returns
    -------
    m_grouped: array
        Matrix with one column per bin, entries summed.
    """
    m_grouped = np.zeros((m.shape[0], len(bins)))
    for i, group in enumerate(bins):
        m_grouped[:, i] = m[:, group].sum(axis=1)
    return m_grouped
