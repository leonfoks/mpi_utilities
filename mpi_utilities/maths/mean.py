import numpy as np
from numba_stdlib import kahan_sum
from ..mpi import Reduce

def mean(values, *, comm, kahan=False, **kwargs):
    """Compute the mean of distributed values.

    Parameters
    ----------
    values : ndarray
        Local array of values.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    mean : float
        The mean of the distributed values.
    """
    N = Reduce(np.size(values), op='sum', comm=comm, **kwargs)

    if N == 0:
        return np.inf

    sm = kahan_sum if kahan else np.sum
    s = Reduce(sm(values), op='sum', comm=comm, **kwargs)
    out = None
    if s is not None:
        out = s / N
    return out