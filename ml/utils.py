"""Utility helpers for ML code: coercions and small adapters used by a few
modules to make types explicit for runtime and static checkers.
"""
from typing import Any
import numpy as np


def to_numpy_array(x: Any, dtype: type = float) -> np.ndarray:
    """Coerce pandas Series/ExtensionArray/iterable to a numpy ndarray with
    a numeric dtype. This centralizes conversions and keeps call sites
    explicit about runtime shapes.
    """
    try:
        arr = np.asarray(x)
    except Exception:
        # Last resort: try converting via list()
        arr = np.array(list(x), dtype=dtype)

    # Ensure numeric dtype
    try:
        return arr.astype(dtype)
    except Exception:
        return np.array(arr, dtype=dtype)
