import numpy as np
from numba_stdlib import kahan_sum as nb_ksum
from ..mpi import Reduce, Gather, Bcast

def kahan_sum(values, *, comm, rank=None, **kwargs):
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
    t_rank = 0 if rank is None else rank

    tmp = Gather(nb_ksum(values), comm=comm, root=t_rank)

    ks = nb_ksum(tmp) if comm.rank == t_rank else None

    return ks if rank is not None else Bcast(ks, comm=comm, root=t_rank)

