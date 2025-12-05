import numpy as np
from .dtype import get_dtype
from ..chunking import load_balance
from .Bcast import Bcast

def Scatterv(values, comm, starts=None, chunks=None, dtype=None, ndim=None, root=0):
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
    if dtype is None:
        dtype = comm.bcast(get_dtype(values, comm, rank=root), root=root)

    if ndim is None:
        # Broadcast the number of dimensions
        ndim = Bcast(np.ndim(values), comm=comm, root=root, ndim=0, dtype='int64')

    if chunks is None:
        if ndim == 1:
            shape = Bcast(np.size(values), comm=comm, root=root, ndim=0, dtype='int64')
        starts, chunks, _ = load_balance(shape, comm.size)
    else:
        if starts is None:
            starts = np.r_[0, np.cumsum(chunks)[:-1]]

    if (ndim == 1):  # For a 1D Array
        out = np.empty(chunks[comm.rank], dtype=dtype)
        comm.Scatterv([values, chunks, starts, None], out, root=root)
        return out