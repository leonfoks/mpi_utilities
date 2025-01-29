import numpy as np
from .common import mpiu_dtype
from .Bcast import Bcast

def Gather(self, world, shape=None, root=0):
    """Gather a numpy array to all ranks in an MPI communicator.

    Each rank gets a chunk such that the size of the array is equally divisible by the number of cores. See the example for more details. Must be called collectively.

    Parameters
    ----------
    self : numpy.ndarray
        A numpy array to broadcast from root.
    dtype : type
        The type of the numpy array being scattered. Must exist on all ranks.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    axis : int, optional
        Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : numpy.ndarray
        A chunk of self on each MPI rank with size chunk[world.rank].

    """
    # if dtype is None:
    dtype = mpiu_dtype(self)

    # if ndim is None:
    # Broadcast the number of dimensions
    ndim = np.ndim(self)

    if ndim == 0:
        shape = world.size
        self = np.atleast_1d(self)

    if (ndim == 1):  # For a 1D Array
        shape = np.size(self) * world.size

    this = None
    if world.rank == root:
        this = np.empty(shape, dtype=dtype)

    world.Gather([self, None], this, root=root)
    return this

    # assert False, ValueError("Gather ndim must equal 1")