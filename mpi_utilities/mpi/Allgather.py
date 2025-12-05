import numpy as np
from .dtype import mpiu_dtype

def Allgather(values, comm):
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
    # if dtype is None:
    dtype = mpiu_dtype(values)

    # if ndim is None:
    # Broadcast the number of dimensions
    ndim = np.ndim(values)

    out = None
    if ndim == 0:
        shape = comm.size
        values = np.atleast_1d(values)

    if (ndim == 1):  # For a 1D Array
        shape = np.size(values) * comm.size

    out = np.empty(shape, dtype=dtype)
    comm.Allgather([values, None], out)
    return out
