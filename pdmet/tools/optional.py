"""
Optional GPU backend support for pdmet

GPU paths in pdmet require cupy, which is *not* a hard dependency because the
cupy wheel that works on a given machine depends on the CUDA toolkit version:

This module gives the rest of pdmet a single place to check for cupy and to
convert arrays between numpy and cupy without sprinkling `try: import cupy`
across every file. CPU-only installations work transparently because cupy
isn't imported at module load time when it's missing.

"""

import numpy as np

# Soft import of cupy, which may not be installed in all environments
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def require_cupy(feature_name="this feature"):
    """Raise a RuntimeError if cupy is not available."""
    if not HAS_CUPY:
        raise RuntimeError(f"{feature_name} requires cupy, which is not installed.")


def is_gpu_mf(mf):
    """Check if the mean-field object is on the GPU."""
    return "gpu4pyscf" in type(mf).__module__


def is_krohf(mf):
    """True if mk is a KROHF, from either pyscf (CPU) or gpu4pyscf (GPU)."""

    from pyscf.pbc.scf.krohf import KROHF as _cpu_KROHF

    if isinstance(mf, _cpu_KROHF):
        return True

    try:
        from gpu4pyscf.pbc.scf.krohf import KROHF as _gpu_KROHF

        if isinstance(mf, _gpu_KROHF):
            return True
    except ImportError:
        pass
    return bool(getattr(mf, "is_ROHF", False))


def to_numpy(x):
    """Recursively convert cupy arrays to numpy. Pass-through for everything else."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_numpy(xi) for xi in x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        return x


def to_cupy(x):
    """Recursively convert numpy arrays to cupy. Pass-through for everything else."""
    if HAS_CUPY and isinstance(x, np.ndarray):
        return cp.asarray(x)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_cupy(xi) for xi in x)
    elif isinstance(x, dict):
        return {k: to_cupy(v) for k, v in x.items()}
    else:
        return x


def get_array_module(*arrays):
    """Return cupy if any input is a cupy array, else numpy.

    The function works on whichever backend the data lives on, without
    duplicating the kernel for CPU and GPU. Most numpy.linalg / numpy.fft /
    numpy.einsum APIs have identical signatures in cupy, so the kernel body
    is usually unchanged.
    """
    if HAS_CUPY:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def _bridge(method, gpu):
    """Wrap a forwarded SCF method so pDMET can call it with numpy inputs
    and receive numpy outputs, even when the underlying method is GPU-backed.

    If gpu=False, returns the method unchanged (zero-cost passthrough).
    If gpu=True, converts every input numpy.ndarray to cupy.ndarray before
    calling (via cupy.asarray), then converts the output back via
    cupy.asnumpy. Non-array arguments (ints, strings, None, etc.) pass
    through both directions untouched.
    """
    if not gpu:
        return method

    def wrapper(*args, **kwargs):
        gpu_args = tuple(to_cupy(a) for a in args)
        gpu_kwargs = {k: to_cupy(v) for k, v in kwargs.items()}
        result = method(*gpu_args, **gpu_kwargs)
        return to_numpy(result)

    return wrapper
