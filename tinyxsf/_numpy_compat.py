"""numpy compatibility shims for old dependencies.

numpy 2.5 removed np.in1d (deprecated since numpy 2.0); cosmolopy and
other old dependencies in the import chain still call it. np.isin is the
drop-in replacement with identical argument order and semantics.
"""
import numpy as np

if not hasattr(np, 'in1d'):
    np.in1d = np.isin
