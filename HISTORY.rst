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

