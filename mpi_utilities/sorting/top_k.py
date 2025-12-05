from copy import copy
from .select import select
from ..mpi.Bcast import Bcast
from ..mpi.Gatherv import Gatherv
from ..mpi.Reduce import Reduce

def top_k(values, k, *, comm, root=None, **kwargs):
    """Compute the top-k values from a distributed array.

    Parameters
    ----------
    values : ndarray
        Local array of values.
    k : int
        Number of top elements to retrieve.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    top_k_values : ndarray
        The top-k values from the distributed array.
    """
    kth = copy(k)
    if k < 0:
        N = Reduce(values.size, op='sum', comm=comm)
        kth = N-k

    # Perform an MPI quickselect to partition the array in-place
    val = select(values, kth - 1, comm=comm, **kwargs)

    local_top_k = values[values >= val]


    t_root = 0 if root is None else root

    all_top_k = Gatherv(local_top_k, root=t_root, comm=comm)

    # Step 3: Global Top-K Selection at root
    top_k_values = None
    if comm.rank == t_root:
        top_k_values.sort()  # Optional: sort the final top-k values

    raise Exception('Not Finished')

    # Step 4: Broadcast the result back to all ranks
    top_k_values = Bcast(top_k_values, root=0, comm=comm)

    return top_k_values
