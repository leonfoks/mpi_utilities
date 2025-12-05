import numpy as np
from numba import jit, int32, int64

_numba_settings = {'nopython': True, 'nogil': False, 'fastmath': True, 'cache': False, 'boundscheck': False}


def best_fit_chunks(shape, n_chunks):
    shape = np.atleast_1d(shape).astype(np.int64)
    n_chunks = np.int64(n_chunks)
    assert n_chunks % 2 == 0, ValueError("n_chunks must be even.")
    return __nb__best_fit_chunks(shape, n_chunks)

# @jit(**_numba_settings)
def __nb__unravel_index(index, shape, order='C'):
    coords = np.ones_like(shape, dtype=np.int32)
    if order == 'C':
        shape = shape[::-1]
    for i, dim in enumerate(shape):
        coords[i] = index % dim
        index //= dim

    if order == 'C':
        coords = coords[::-1]
    return coords

# @jit(int64[:](int64[:], int64), **_numba_settings)
def __nb__best_fit_chunks(shape, n_chunks):
    n_dim = shape.size

    weights = shape / np.max(shape)
    ordering = np.argsort(weights)
    sorted_weights = weights[ordering]

    limits = np.asarray(n_chunks / np.arange(n_dim, 0, -1), dtype=np.int32)

    k = n_chunks-1
    chunks = __nb__unravel_index(k, np.full(n_dim, fill_value=n_chunks)) + 1
    best_fit = 1e20

    test = chunks
    go = True
    while go:
        if np.prod(test) == n_chunks and np.all(test <= limits):
            fit = np.linalg.norm(sorted_weights - (test / np.max(test)))

            if fit < best_fit:
                chunks = test
                best_fit = fit

        k += 1
        test = __nb__unravel_index(k, np.full(n_dim, fill_value=n_chunks)) + 1

        go = test[0] <= limits[0]

    return chunks[np.argsort(ordering)]