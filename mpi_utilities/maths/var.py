import numpy as np
from numba_stdlib import kahan_sum
from ..mpi import Reduce

def var(values, *, comm, **kwargs):
    """Compute the median of distributed values.

    Parameters
    ----------
    values : ndarray
        Local array of values.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    median : float
        The median of the distributed values.
    """

    ddof = kwargs.pop('ddof', 1)

    N = Reduce(np.size(values), op='sum', comm=comm)

    if N <= 1:
        return np.inf
    s = Reduce(kahan_sum(values), op='sum', comm=comm)
    mean = s / N

    s = Reduce(kahan_sum((values - mean)**2.0), op='sum', comm=comm, **kwargs)

    out = None
    if s is not None:
        out = s / (N - ddof)
    return out


