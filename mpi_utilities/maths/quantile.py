import numpy as np
from ..mpi import Allreduce
from ..sorting import select

def quantile(values, q, *, comm, **kwargs):
    """Compute the quantile of distributed values.

    Parameters
    ----------
    values : ndarray
        Local array of values.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    quantile : float
        The quantile of the distributed values.
    """
    assert 0 < q < 1, ValueError("Quantile q must be in the range (0, 1)")
    N = Allreduce(np.size(values), op='sum', comm=comm)

    k = np.int64(q * (N - 1))

    return select(values, k, comm=comm, **kwargs)