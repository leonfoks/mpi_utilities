import numpy as np
from numba_stdlib import kahan_sum
from ..mpi import Reduce

def norm(values, *, comm, ord=2.0, kahan=False, **kwargs):
    """Compute the norm of distributed values.

    Parameters
    ----------
    values : ndarray
        Local array of values.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    norm : float
        The norm of the distributed values.
    """

    method = kahan_sum if kahan else np.sum

    root_me = False
    match ord:
        case 0:
            sm = np.sum(values != 0.0)
        case 1:
            sm = method(np.abs(values))
        case np.inf:
            sm = np.max(np.abs(values))
        case _:
            root_me = True
            sm = method(np.power(values, ord))

    if ord == np.inf:
        sm = Reduce(sm, op='max', comm=comm, **kwargs)
    else:
        sm = Reduce(sm, op='sum', comm=comm, **kwargs)

    if root_me:
        sm = np.power(sm, 1.0/ord)
    return sm