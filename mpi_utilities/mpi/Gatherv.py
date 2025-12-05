import numpy as np
from .dtype import mpiu_dtype
from .Gather import Gather
from .Allgather import Allgather
from .Allgatherv import Allgatherv

def Gatherv(values, comm, starts=None, chunks=None, root=None, **kwargs):
    """ScatterV a numpy array to all ranks in an MPI communicator.

    Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. See the example for more details. Must be called collectively.

    Parameters
    ----------
    values : numpy.ndarray
        A numpy array to broadcast from root.
    starts : array of ints
        1D array of ints with size equal to the number of MPI ranks.
        Each element gives the starting index for a chunk to be sent to that core.
        e.g. starts[0] is the starting index for rank = 0.
        Must exist on all ranks
    chunks : array of ints
        1D array of ints with size equal to the number of MPI ranks.
        Each element gives the size of a chunk to be sent to that core.
        e.g. chunks[0] is the chunk size for rank = 0.
        Must exist on all ranks
    dtype : type
        The type of the numpy array being scattered. Must exist on all ranks.
    comm : mpi4py.MPI.Comm
        MPI parallel communicator.
    axis : int, optional
        Axis along which to Scatterv to the ranks if values is a 2D numpy array. Default is 0
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : numpy.ndarray
        A chunk of values on each MPI rank with size chunk[comm.rank].

    """

    if root is None:
        return Allgatherv(values, comm=comm, starts=starts, chunks=chunks)

    # if dtype is None:
    dtype = mpiu_dtype(values)
    # comm.bcast(get_dtype(values, comm, rank=root), root=root)

    # if ndim is None:
        # Broadcast the number of dimensions
        # ndim = Bcast(np.ndim(values), comm, root=root, ndim=0, dtype='int64')
    ndim = np.ndim(values)

    if ndim == 0:
        return Gather(values, comm=comm, root=root)

    this = None
    if ndim == 1:
        if chunks is None:
            chunks = Allgather(np.size(values), comm=comm)

        if starts is None:
            starts = np.hstack([0, np.cumsum(chunks[:-1])])

        if comm.rank == root:
            this = np.empty(np.sum(chunks), dtype=dtype)

        comm.Gatherv(values, [this, chunks, starts, None], root=root)
        return this