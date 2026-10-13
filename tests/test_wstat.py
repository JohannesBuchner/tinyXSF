"""Tests for the W statistic (profile Poisson likelihood with Poisson background).

The reference is the definition of the wstat statistic (see
https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node393.html):
the per-bin background amplitude is profiled out of the joint
source+background Poisson likelihood. These tests verify the analytic
solution against a numerical maximization of that likelihood.
"""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import tinyxsf

# wstat warns when a bin has fewer than 3 background counts; most tests
# exercise that regime deliberately, so silence the warning here
pytestmark = pytest.mark.filterwarnings('ignore:wstat is biased')


def joint_loglikelihood(m_rate, S, B, ts, tb, f):
    """Joint source+background Poisson log-likelihood of one bin.

    m_rate is the source model rate m_i (model counts in the source
    region divided by ts), S and B the observed source and
    background-region counts, ts and tb the exposure*areascal of the
    source and background spectra, and f the free background amplitude
    in source-region counts (source region expects ts*(m+f), background
    region tb*f). Factorial terms depending only on the data are
    neglected.
    """
    # the -tb*f term is the probability of the background-region counts
    # and applies even when B == 0 (observing zero counts given tb*f
    # expected has probability exp(-tb*f))
    like = S * np.log(ts * (m_rate + f)) - ts * (m_rate + f) - tb * f
    if B > 0:
        like = like + B * np.log(tb * f)
    return like


def saturated_loglikelihood(S, B):
    """Maximum likelihood of the saturated model (free m and f per bin)."""
    like = 0.0
    if S > 0:
        like += S * np.log(S) - S
    if B > 0:
        like += B * np.log(B) - B
    return like


def numerical_wstat(m_rate, S, B, ts, tb):
    """W statistic of one bin from a numerical profile over the background."""
    import scipy.optimize

    if S + B > 0:
        fhi = 2 * (S + B) / (ts + tb) + 1
    else:
        fhi = 1.0
    res = scipy.optimize.minimize_scalar(
        lambda f: -joint_loglikelihood(m_rate, S, B, ts, tb, f),
        bounds=(1e-12, fhi), method='bounded', options={'xatol': 1e-13})
    return 2 * (saturated_loglikelihood(S, B) + res.fun)


def random_bins(rng, n):
    """Random bin contents covering the special cases (zero counts)."""
    S = rng.integers(0, 100, n).astype(float)
    B = rng.integers(0, 3000, n).astype(float)
    M = rng.uniform(0, 100, n)
    ts = rng.uniform(1e3, 1e6, n)
    tb = rng.uniform(1e3, 1e6, n)
    ratio = rng.uniform(0.1, 30, n)
    # make sure all special cases appear
    S[:10] = 0
    B[10:20] = 0
    S[20:30] = 0
    B[20:30] = 0
    M[:5] = 0
    return S, B, M, ts, tb, ratio


def test_wstat_matches_numerical_profile():
    rng = np.random.default_rng(42)
    n = 200
    S, B, M, ts, tb, ratio = random_bins(rng, n)
    expected = np.array([
        # wstat takes model counts (t_s*m); the likelihood works with the rate
        numerical_wstat(M[i] / ts[i], S[i], B[i], ts[i], tb[i] * ratio[i])
        for i in range(n)])
    computed = np.array([
        tinyxsf.wstat(model=M[i:i + 1], src_counts=S[i:i + 1],
                      bkg_counts=B[i:i + 1], src_expoarea=ts[i],
                      bkg_expoarea=tb[i], backscale_ratio=ratio[i])
        for i in range(n)])
    assert_allclose(computed, expected, rtol=1e-4, atol=1e-4)


def test_wstat_special_cases():
    ts, tb, ratio = 100.0, 400.0, 2.0
    tbr = tb * ratio

    # S = B = 0: W_i = 2 t_s m_i
    assert_allclose(tinyxsf.wstat(model=[7.0], src_counts=[0], bkg_counts=[0],
                                  src_expoarea=ts, bkg_expoarea=tb), 14.0)
    # S = 0, B > 0: f_i = B_i/(t_s+t_b), W_i = 2 (t_s m_i - B_i ln(t_b/(t_s+t_b)))
    S, B, M = 0.0, 30.0, 5.0
    expected = 2 * (M - B * np.log(tbr / (ts + tbr)))
    assert_allclose(tinyxsf.wstat(model=[M], src_counts=[S], bkg_counts=[B],
                                  src_expoarea=ts, bkg_expoarea=tb,
                                  backscale_ratio=ratio), expected)
    # B = 0, m_i < S_i/(t_s+t_b): f_i = S_i/(t_s+t_b) - m_i,
    # W_i = 2 (-t_b m_i - S_i ln(t_s/(t_s+t_b)))
    S, M = 100.0, 2.0
    assert M / ts < S / (ts + tbr)
    expected = 2 * (-tbr * M / ts - S * np.log(ts / (ts + tbr)))
    assert_allclose(tinyxsf.wstat(model=[M], src_counts=[S], bkg_counts=[0],
                                  src_expoarea=ts, bkg_expoarea=tb,
                                  backscale_ratio=ratio), expected)
    # B = 0, m_i >= S_i/(t_s+t_b): f_i = 0, W_i is the Poisson deviance
    S, M = 100.0, 20.0
    assert M / ts >= S / (ts + tbr)
    expected = 2 * (M + S * (np.log(S) - np.log(M) - 1))
    assert_allclose(tinyxsf.wstat(model=[M], src_counts=[S], bkg_counts=[0],
                                  src_expoarea=ts, bkg_expoarea=tb,
                                  backscale_ratio=ratio), expected)


def test_wstat_warns_for_few_background_counts():
    # the warning is silenced module-wide; pytest.warns and
    # simplefilter('error') re-enable recording/capture
    with pytest.warns(RuntimeWarning, match='biased'):
        tinyxsf.wstat(model=[10.0], src_counts=[5], bkg_counts=[1],
                      src_expoarea=100.0, bkg_expoarea=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tinyxsf.wstat(model=[10.0], src_counts=[5], bkg_counts=[30],
                      src_expoarea=100.0, bkg_expoarea=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tinyxsf.wstat_vectorized(models=np.full(100, 10.0),
                                 src_counts=np.full(100, 5),
                                 bkg_counts=np.full(100, 30),
                                 src_expoarea=100.0, bkg_expoarea=400.0)


def test_wstat_nonnegative():
    rng = np.random.default_rng(43)
    S, B, M, ts, tb, ratio = random_bins(rng, 100)
    w = np.array([
        tinyxsf.wstat(model=M[i:i + 1], src_counts=S[i:i + 1],
                      bkg_counts=B[i:i + 1], src_expoarea=ts[i],
                      bkg_expoarea=tb[i], backscale_ratio=ratio[i])
        for i in range(100)])
    assert np.all(w >= 0)


def test_wstat_vectorized_matches_scalar():
    rng = np.random.default_rng(44)
    S, B, M, ts, tb, ratio = random_bins(rng, 50)
    vectorized = tinyxsf.wstat_vectorized(
        models=M.reshape(1, -1), src_counts=S, bkg_counts=B,
        src_expoarea=ts.mean(), bkg_expoarea=tb.mean())
    scalar = tinyxsf.wstat(model=M, src_counts=S, bkg_counts=B,
                           src_expoarea=ts.mean(), bkg_expoarea=tb.mean())
    assert_allclose(vectorized, [scalar])


def test_wstat_vectorized_rowwise():
    # each row of models is evaluated independently
    rng = np.random.default_rng(45)
    n = 30
    S, B, M, ts, tb, ratio = random_bins(rng, n)
    rows = tinyxsf.wstat_vectorized(
        models=np.stack([M, 2 * M]), src_counts=S, bkg_counts=B,
        src_expoarea=ts.mean(), bkg_expoarea=tb.mean())
    assert rows.shape == (2,)
    # the vectorized result is the per-bin sum for each model row
    assert_allclose(rows[0], sum(
        tinyxsf.wstat(model=M[i:i + 1], src_counts=S[i:i + 1],
                      bkg_counts=B[i:i + 1], src_expoarea=ts.mean(),
                      bkg_expoarea=tb.mean())
        for i in range(n)))
    assert_allclose(rows[1], sum(
        tinyxsf.wstat(model=2 * M[i:i + 1], src_counts=S[i:i + 1],
                      bkg_counts=B[i:i + 1], src_expoarea=ts.mean(),
                      bkg_expoarea=tb.mean())
        for i in range(n)))


def test_wstat_per_bin_profile_example_magnitudes():
    # same as test_wstat_matches_numerical_profile but with the
    # count magnitudes and exposure ratio of a faint-source observation
    rng = np.random.default_rng(46)
    n = 100
    ts = 1e6
    tb = 32.4418e6
    S = rng.poisson(1.0, n).astype(float)
    B = rng.poisson(30.0, n).astype(float)
    M = rng.uniform(0, 20, n)
    expected = np.array([
        numerical_wstat(M[i] / ts, S[i], B[i], ts, tb) for i in range(n)])
    computed = np.array([
        tinyxsf.wstat(model=M[i:i + 1], src_counts=S[i:i + 1],
                      bkg_counts=B[i:i + 1], src_expoarea=ts,
                      bkg_expoarea=tb)
        for i in range(n)])
    assert_allclose(computed, expected, rtol=1e-4, atol=1e-4)
    # the vectorized call returns the W summed over bins
    assert_allclose(
        tinyxsf.wstat_vectorized(models=M.reshape(1, -1), src_counts=S,
                                 bkg_counts=B, src_expoarea=ts,
                                 bkg_expoarea=tb)[0],
        expected.sum(), rtol=1e-4, atol=1e-4)


def test_wstat_minimum_near_dof():
    # fitting the true model to data drawn from it gives W ~ n_bins - n_free
    rng = np.random.default_rng(47)
    nbins = 200
    ts, tb = 1e5, 4e5
    m_counts_true = rng.uniform(1, 10, nbins)
    bkg_counts_true = rng.uniform(5, 50, nbins)
    S = rng.poisson(m_counts_true + ts * bkg_counts_true / tb)
    B = rng.poisson(bkg_counts_true)
    # one free parameter: scaling of the true model
    import scipy.optimize

    def wstat_of_scale(logscale):
        return tinyxsf.wstat(
            model=m_counts_true * np.exp(logscale), src_counts=S,
            bkg_counts=B, src_expoarea=ts, bkg_expoarea=tb)

    res = scipy.optimize.minimize_scalar(
        wstat_of_scale, bounds=(-2, 2), method='bounded')
    # W is approximately chi^2 with nbins - 1 degrees of freedom
    assert abs(res.fun - (nbins - 1)) < 5 * np.sqrt(2 * (nbins - 1))
    assert abs(np.exp(res.x) - 1) < 0.2
