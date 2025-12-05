import numpy as np
from .dtype import mpiu_dtype
from .Allgather import Allgather

def Gather(values, *, comm, out=None, root=None):
    """Gather a numpy array to all ranks in an MPI communicator.

    Each rank gets a chunk such that the size of the array is equally divisible by the number of cores. See the example for more details. Must be called collectively.

    Parameters
    ----------
    values : numpy.ndarray
        A numpy array to broadcast from root.
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
        return Allgather(values, comm=comm)

    # if dtype is None:
    dtype = mpiu_dtype(values)

    # if ndim is None:
    # Broadcast the number of dimensions
    ndim = np.ndim(values)

    if ndim == 0:
        shape = comm.size
        values = np.atleast_1d(values)

    if (ndim == 1):  # For a 1D Array
        shape = np.size(values) * comm.size

    this = None
    if comm.rank == root:
        this = np.empty(shape, dtype=dtype) if out is None else out

    comm.Gather([values, None], this, root=root)
    return this