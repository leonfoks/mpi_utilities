import itertools
import numpy as np
from ..numba_methods import best_fit_chunks

def starts_from_chunks(values):
    return np.r_[0, np.cumsum(values[:-1])]

def cartesian_chunks(shape, *, starts=None, increment=None, **kwargs):
    """Generate a cartesian iterable over axes"""
    shape = np.atleast_1d(shape)
    increment = np.atleast_1d(increment)

    iterables = []
    if starts is None:
        starts = np.zeros(np.size(shape), dtype=np.int64)
    if increment is None:
        increment = np.ones(np.size(shape), dtype=np.int54)

    for i in range(np.size(shape)):
        inc = increment[i]
        tmp = np.arange(start=starts[i], stop=shape[i]+inc, step=inc, dtype=np.int64)
        tmp[-1] = np.minimum(tmp[-1], shape[i])
        iterables.append(zip(tmp[:-1], tmp[1:]))
    # starts = np.stack(np.meshgrid(*s1d, indexing='ij'), axis=-1).reshape(-1, len(s1d))
    # chunks = np.stack(np.meshgrid(*c1d, indexing='ij'), axis=-1).reshape(-1, len(c1d))
    # return itertools.product(*iterable_1d)
    return iterables

def load_balance(shape, n_chunks, flatten=True):

    if (np.ndim(shape) == 0) or ((np.ndim(shape) == 1) and (np.size(shape) == 1)):
        s, c = load_balance_1d(shape, n_chunks)
        return s, c, np.r_[n_chunks]

    bf = best_fit_chunks(shape, n_chunks)

    s1d = []; c1d = []
    for i in range(np.size(shape)):
        s, c = load_balance_1d(shape[i], bf[i])
        s1d.append(s); c1d.append(c)

    # Use itertools.product to get all combinations
    # The '*' unpacks the list of arrays as separate arguments to product
    starts = np.stack(np.meshgrid(*s1d, indexing='ij'), axis=-1).reshape(-1, len(s1d))
    chunks = np.stack(np.meshgrid(*c1d, indexing='ij'), axis=-1).reshape(-1, len(c1d))

    if not flatten:
        starts = starts.reshape(*bf, np.size(shape))
        chunks = chunks.reshape(*bf, np.size(shape))

    return starts, chunks, bf

def load_balance_1d(N, n_chunks):
    """Splits the length of an array into a number of chunks. Load balances the chunks in a shrinking arrays fashion.

    Given length, N, split N up into n_chunks and return the starting index and size of each chunk.
    After being split equally among the chunks, the remainder is distributed so that chunks 0:remainder
    get +1 in size. e.g. N=10, n_chunks=3 would return starts=[0,4,7] chunks=[4,3,3]

    Parameters
    ----------
    N : int
        A size to split into chunks.
    n_chunks : int
        The number of chunks to split N into. Usually the number of ranks, comm.size.

    Returns
    -------
    starts : ndarray of ints
        The starting indices of each chunk.
    chunks : ndarray of ints
        The size of each chunk.

    """
    N = np.asarray(N).item()
    chunks = np.full(n_chunks, fill_value=N/n_chunks, dtype=np.int64)
    mod = np.int64(N % n_chunks)
    chunks[:mod] += 1
    starts = np.cumsum(chunks) - chunks[0]
    if (mod > 0):
        starts[mod:] += 1
    return starts, chunks