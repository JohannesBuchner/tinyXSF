TinyXSF
-------

Fast X-ray spectral fitting.

.. image:: https://img.shields.io/pypi/v/tinyxsf.svg
        :target: https://pypi.python.org/pypi/tinyxsf

.. image:: https://github.com/JohannesBuchner/tinyxsf/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/JohannesBuchner/tinyxsf/actions/workflows/tests.yml

.. image:: https://coveralls.io/repos/github/JohannesBuchner/tinyxsf/badge.svg?branch=main
	:target: https://coveralls.io/github/JohannesBuchner/tinyxsf?branch=main

.. image:: https://img.shields.io/badge/GitHub-JohannesBuchner%2Ftinyxsf-blue.svg?style=flat
        :target: https://github.com/JohannesBuchner/tinyxsf/
        :alt: Github repository

Background
----------

Currently, there are the following issues in modern X-ray spectral fitting software:

1. Response matrices have become huge (e.g. XRISM, NuSTAR), making models slow to evaluate.
2. XSPEC is not developed openly and its quirks make it difficult to build upon it and extend it.
3. Yet models are maintained by the community in XSPEC.
4. Maintaining additional software packages requires substantial institutional efforts (CXC: sherpa, SRON: spex).
5. Not all models are differentiable. Reimplementing them in a differentiable language one by one is a significant effort with little recognition.
   Nevertheless, it has been and is being tried. Yet such reimplementations tend to fade out (see also 3ML astromodels).
6. Inference parameter spaces are complicated, with multiple modes and other complicated degeneracies being common in X-ray spectral fitting.
7. `Bayesian model comparison <https://ui.adsabs.harvard.edu/abs/2014A%26A...564A.125B/>`_ is powerful and we want it.
8. The X-ray community is walled off from other communities by vendor lock-in and its own odd terminology. But `X-ray spectral fitting is neither complicated nor a special case! <https://arxiv.org/abs/2309.05705>`_

Therefore, we want:

1) A performant software package
2) All community packages from XSPEC
3) Nested sampling for model comparison and robust parameter estimation
4) Minimum package maintainance effort

TinyXSF does that.

xspex&jaxspec do 1, xspec/sherpa+BXA does 2+3.

TinyXSF is a few hundred lines of code.

Approach
--------

1) Vectorization.
   Folding the spectrum through the RMF is vectorized.
   Handling many proposed spectra at once keeps memory low and efficiency high:
   Each chunk of the response matrix is applied to all spectra.
   Modern Bayesian samplers such as UltraNest can handle vectorized functions.

2) Building upon the CXC (Doug Burke's) wrapper for Xspec models. https://github.com/cxcsds/xspec-models-cxc/
   All XSPEC models are available for use!

3) Some further niceties (all optional) include handling of backgrounds, redshifts and galactic NH:

   * Use BXA's autobackground folder to create a background spectral model from your background region.
   * Use BXA's galnh.py to fetch the galactic NH for the position of your observation and store it in my.pha.nh as a string (e.g. 1.2e20).
   * Store the redshift in my.pha.z as a string.

4) We treat X-ray spectral fitting as a normal inference problem like any other!

   Define a likelihood, prior and call a sampler. No need to carry around
   legacy awkwardness such as chi-square, C-stat, 
   background-subtraction, identify matrix folding, multi-source to data mappings.

Installation
------------

Quick version with conda/mamba::

    $ conda create -n tinyXSF -c https://cxc.cfa.harvard.edu/conda/ciao -c xspec-modelsonly conda-forge astropy ultranest jax scikit-learn 
    $ pip install git+https://github.com/cxcsds/xspec-models-cxc --verbose
    $ pip install git+https://github.com/JohannesBuchner/tinyXSF



Prerequisites:

* install and load xspec/heasoft
* install https://github.com/cxcsds/xspec-models-cxc/
* install ultranest (pip install ultranest)
* download this repository and enter it from the command line.

Test with::

   `python -c 'import xspec_models_cxc; import ultranest'`

Getting started
---------------

To start, have a look at simple.py, which demonstrates:

* loading a spectrum
* loading a ATable (download the table from `the xars models page <https://github.com/JohannesBuchner/xars/blob/master/doc/README.rst>`_)
* setting up a XSPEC model
* passing the model through the ARF and RMF
* adding a background model
* plotting the spectrum of the source and background model on top of the data
* computing the likelihood and print it

Next, the vectorization is in simplev.py, which demonstrates the same as above plus:

* vectorized handling of proposals
* launching UltraNest for sampling the posterior, make corner plots.

Next, there is simpleopt.py, which demonstrates optimized nested sampling (optNS).
This is much faster, especially when there are many components with free normalisations.

Take it for a spin and adapt it!

Todo
----

* ✓ Profile where the code is still slow.
* ✓ There is a python loop in tinyxsf/model.py::Table.__call__ which should be replaced with something smarter
* ✓ Compute fluxes and luminosities.
* ✓ Create some unit tests for loading and evaluating atables/mtables, poisson probability, plotting ARF/RMF.
* ✓ Make a atable that is precomputed for a given rest-frame energy grid at fixed redshift. -> FixedTable

Credits
--------

This builds upon work by the Chandra X-ray Center (in particular Doug Burke's wrapper),
and Daniela Huppenkothen's RMF/ARF reader (based in turn on sherpa code, IIRC).

License: GPL v3

Contributions are welcome.
