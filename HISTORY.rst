1.2.3
-----
* Add ``tinyxsf.wstat`` and ``tinyxsf.wstat_vectorized``: the W statistic
  (XSPEC wstat), a profile Poisson likelihood with Poisson-distributed
  background that requires no background model. The per-bin background
  amplitude is profiled out analytically; special cases for bins with
  zero source or background counts are handled. The arguments are
  keyword-only. A warning is emitted when a bin has fewer than 3
  background counts, where the profile-likelihood background treatment
  is biased (regroup with e.g. ``ftgrouppha grouptype=bmin groupscale=3``).
  ``load_pha`` now also returns the ``backscal`` values
  (``src_backscal``, ``bkg_backscal``, ``backscale_ratio``) needed to
  scale the background region. Verified against XSPEC
  (``wstat_xspec_check.py``) and against a numerical maximization of
  the joint source+background Poisson likelihood
  (``tests/test_wstat.py``).

1.2.2
-----
* Apply the OGIP GROUPING column of the source spectrum on loading:
  source counts, background counts, the RMF (matrix entries of grouped
  channels summed) and the background model sidecar are combined per
  group; QUALITY-flagged and 0-marked channels are ignored. The
  grouping logic lives in the new ``tinyxsf.grouping`` module.

1.2.1
-----
* Add ``tinyxspec`` command: set up an XSPEC-style model expression,
  manage per-parameter priors in ``parameters.txt`` and sample the priors
  to plot the resulting spectra.

1.0.0
-----
First version

