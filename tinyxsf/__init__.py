"""Tiny X-ray spectral fitting library."""

from . import _numpy_compat  # noqa: F401  (numpy compat shims, import first)

__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '1.2.3'

from .data import load_pha
from .model import (FixedTable, Table, logNegBinomialPDF,
                    logNegBinomialPDF_vectorized, logPoissonPDF,
                    logPoissonPDF_vectorized, wstat, wstat_vectorized, x, xvec)
