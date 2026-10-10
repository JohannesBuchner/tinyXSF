"""Tests for OGIP grouping support in load_pha."""
import os
import shutil

import astropy.io.fits as pyfits
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import tinyxsf
from tinyxsf.data import get_ARF, get_RMF
from tinyxsf.grouping import parse_grouping

example_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'example'))

SIDECARS = ['cdfs4Ms_179.rmf', 'cdfs4Ms_179.arf',
            'cdfs4Ms_179_bkg.pi', 'cdfs4Ms_179_bkg.pi_model.fits']


def clear_response_cache():
    # load_pha strips the (cached) RMF objects; start fresh per file
    get_RMF.cache_clear()
    get_ARF.cache_clear()


def write_grouped_pha(tmp_path, grouping=None, quality=None):
    """Write a copy of the example spectrum with an added GROUPING column."""
    src = os.path.join(example_dir, '179.pi')
    dst = str(tmp_path / 'grouped.pi')
    with pyfits.open(src) as a:
        header = a["SPECTRUM"].header.copy()
        d = a["SPECTRUM"].data.copy()
    # the example file marks no grouping with a GROUPING keyword of 0
    # (OGIP); grouping is defined by the column added below
    header.pop('GROUPING', None)
    columns = [pyfits.Column(name='CHANNEL', format='I', array=d['CHANNEL']),
               pyfits.Column(name='COUNTS', format='J', array=d['COUNTS'])]
    if grouping is not None:
        columns.append(pyfits.Column(name='GROUPING', format='I',
                                     array=np.asarray(grouping, dtype=np.int16)))
    if quality is not None:
        columns.append(pyfits.Column(name='QUALITY', format='I',
                                     array=np.asarray(quality, dtype=np.int16)))
    hdu = pyfits.BinTableHDU.from_columns(columns, header=header)
    pyfits.HDUList([pyfits.PrimaryHDU(), hdu]).writeto(dst)
    for sidecar in SIDECARS:
        shutil.copy(os.path.join(example_dir, sidecar),
                    os.path.join(str(tmp_path), sidecar))
    return dst


def read_native_counts(name):
    with pyfits.open(os.path.join(example_dir, name)) as a:
        return np.array(a['SPECTRUM'].data['COUNTS'])


def test_parse_grouping_standard():
    # +1 opens a group, -1 continues it, 0 marks unused channels
    grouping = np.array([1, -1, -1, 1, 1, -1, 0, 1])
    groups = parse_grouping(grouping)
    assert [list(g) for g in groups] == [[0, 1, 2], [3], [4, 5], [7]]


def test_parse_grouping_single_positive_values():
    grouping = np.array([1, 1, 1])
    assert [list(g) for g in parse_grouping(grouping)] == [[0], [1], [2]]


def test_parse_grouping_end_marked():
    # some tools mark the last channel of each group with 1 (members -1)
    grouping = np.array([-1, -1, 1, -1, 1, 1, 0, 1])
    groups = parse_grouping(grouping)
    assert [list(g) for g in groups] == [[0, 1, 2], [3, 4], [5], [7]]


def test_parse_grouping_invalid():
    with pytest.raises(ValueError):
        parse_grouping(np.array([1, 2, 1]))


def test_load_ungrouped_unchanged():
    # a spectrum without GROUPING column loads as before
    clear_response_cache()
    data = tinyxsf.load_pha(os.path.join(example_dir, '179.pi'), 0.5, 8.0)
    assert data['grouping'] is None
    assert data['Nchan'] == 512
    assert data['chan_mask'].shape == (1024,)
    assert data['chan_mask'].sum() == 512


def test_load_ftgrouppha_file():
    # a file grouped with ftgrouppha (grouptype=bmin groupscale=3 on the
    # background counts), loaded with a band that contains all groups
    clear_response_cache()
    elo, ehi = 0.1, 14.95
    data = tinyxsf.load_pha(os.path.join(example_dir, '179_g3.pi'), elo, ehi)
    # both files share the same RMF file, which load_pha strips in place
    clear_response_cache()
    data_u = tinyxsf.load_pha(os.path.join(example_dir, '179.pi'), elo, ehi)
    groups = data['grouping']

    assert len(groups) == data['Nchan']
    # groups partition exactly the in-band channels
    assert_array_equal(np.sort(np.concatenate(groups)),
                       np.flatnonzero(data_u['chan_mask']))
    # some multi-channel groups exist in the band
    assert len(set(len(g) for g in groups)) > 1
    for g in groups:
        assert len(g) > 0 and np.all(np.diff(g) > 0)

    # counts are the sums over each group
    counts = read_native_counts('179.pi')
    bcounts = read_native_counts('cdfs4Ms_179_bkg.pi')
    assert_array_equal(data['src_region_counts'],
                       np.array([counts[g].sum() for g in groups]))
    assert_array_equal(data['bkg_region_counts'],
                       np.array([bcounts[g].sum() for g in groups]))
    assert data['src_region_counts'].sum() == data_u['src_region_counts'].sum()

    # channel edges are the outermost of each group
    ebounds = pyfits.getdata(os.path.join(example_dir, 'cdfs4Ms_179.rmf'), 'EBOUNDS')
    assert_allclose(data['chan_e_min'], [ebounds['E_MIN'][g].min() for g in groups])
    assert_allclose(data['chan_e_max'], [ebounds['E_MAX'][g].max() for g in groups])

    # the dense RMF matrix has one column per group with summed entries
    m_u = data_u['RMF']
    assert data['RMF'].shape == (m_u.shape[0], len(groups))
    assert_allclose(data['RMF'].sum(), m_u.sum(), rtol=1e-12)

    # energy grid and ARF are per energy bin and unchanged by grouping
    assert_allclose(data['e_lo'], data_u['e_lo'])
    assert_allclose(data['e_hi'], data_u['e_hi'])
    assert_allclose(data['ARF'], data_u['ARF'])
    # chan_mask is a trivial mask over the grouped bins
    assert data['chan_mask'].shape == (len(groups),) and data['chan_mask'].all()


def test_load_grouped_folding(tmp_path):
    # folding a model through the grouped RMF equals folding per channel
    # and summing each group; sidecar background model grouped likewise
    clear_response_cache()
    elo, ehi = 0.5, 8.0
    data_u = tinyxsf.load_pha(os.path.join(example_dir, '179.pi'), elo, ehi)
    mask = data_u['chan_mask']
    inband = np.flatnonzero(mask)

    # hand-made grouping: pairs for the first 40 in-band channels,
    # one channel marked unused, rest single-channel groups
    grouping = np.ones(1024, dtype=int)
    for c in inband[:40:2]:
        grouping[c] = 1
        grouping[c + 1] = -1
    grouping[inband[60]] = 0
    fn = write_grouped_pha(tmp_path, grouping=grouping)

    data = tinyxsf.load_pha(fn, elo, ehi)
    groups = data['grouping']
    assert len(groups) == data['Nchan']

    # groups cover the in-band channels except the 0-marked one
    assert_array_equal(np.sort(np.concatenate(groups)),
                       np.setdiff1d(inband, [inband[60]]))
    # first 20 groups are the pairs
    assert [len(g) for g in groups[:20]] == [2] * 20

    counts = read_native_counts('179.pi')
    bcounts = read_native_counts('cdfs4Ms_179_bkg.pi')
    assert_array_equal(data['src_region_counts'],
                       np.array([counts[g].sum() for g in groups]))
    assert_array_equal(data['bkg_region_counts'],
                       np.array([bcounts[g].sum() for g in groups]))
    # the dropped channel is excluded from the totals
    assert data['src_region_counts'].sum() == (
        data_u['src_region_counts'].sum() - counts[inband[60]])

    # folding a test spectrum: grouped RMF matches per-channel fold + group sums
    energies = data_u['energies']
    spec = energies[:-1] ** -2.0
    folded_perchan = np.asarray(
        data_u['RMF_src'].apply_rmf(spec * data_u['ARF']))[mask]
    chan_index = np.full(1024, -1)
    chan_index[mask] = np.arange(mask.sum())
    folded_manual = np.array(
        [folded_perchan[chan_index[g]].sum() for g in groups])
    folded = np.asarray(
        data['RMF_src'].apply_rmf(spec * data['ARF']))[data['chan_mask']]
    assert_allclose(folded, folded_manual, rtol=1e-5)

    # vectorized folding path
    folded_vec = data['RMF_src'].apply_rmf_vectorized(
        np.asarray(spec * data['ARF'])[None, :])[0][data['chan_mask']]
    assert_allclose(folded_vec, folded_manual, rtol=1e-5)

    # background model sidecar (per-channel INTPSPEC) grouped likewise
    bkg_model = pyfits.getdata(
        os.path.join(str(tmp_path), 'cdfs4Ms_179_bkg.pi_model.fits'), 'SPECTRA')
    bm_src = np.array(bkg_model[1]['INTPSPEC'])
    bm_bkg = np.array(bkg_model[0]['INTPSPEC'])
    assert_allclose(data['bkg_model_src_region'], [bm_src[g].sum() for g in groups])
    assert_allclose(data['bkg_model_bkg_region'], [bm_bkg[g].sum() for g in groups])


def test_load_grouped_quality(tmp_path):
    # QUALITY-flagged channels are excluded from the bins
    clear_response_cache()
    elo, ehi = 0.5, 8.0
    data_u = tinyxsf.load_pha(os.path.join(example_dir, '179.pi'), elo, ehi)
    mask = data_u['chan_mask']
    inband = np.flatnonzero(mask)

    grouping = np.ones(1024, dtype=int)
    grouping[inband[0] + 1] = -1  # one pair
    quality = np.zeros(1024, dtype=int)
    quality[inband[5]] = 5
    quality[inband[7]] = 2
    fn = write_grouped_pha(tmp_path, grouping=grouping, quality=quality)

    data = tinyxsf.load_pha(fn, elo, ehi)
    groups = data['grouping']
    covered = np.sort(np.concatenate(groups))
    assert inband[5] not in covered and inband[7] not in covered
    assert_array_equal(covered, np.setdiff1d(inband, [inband[5], inband[7]]))

    counts = read_native_counts('179.pi')
    assert data['src_region_counts'].sum() == (
        data_u['src_region_counts'].sum() - counts[[inband[5], inband[7]]].sum())


def test_load_grouping_keyword_zero(tmp_path):
    # a GROUPING keyword of 0 means no grouping is defined (OGIP):
    # the column is ignored
    clear_response_cache()
    elo, ehi = 0.5, 8.0
    grouping = np.ones(1024, dtype=int)
    grouping[1:40] = -1
    fn = write_grouped_pha(tmp_path, grouping=grouping)
    with pyfits.open(fn, mode='update') as a:
        a['SPECTRUM'].header['GROUPING'] = 0

    data = tinyxsf.load_pha(fn, elo, ehi)
    assert data['grouping'] is None
    assert data['Nchan'] == 512
