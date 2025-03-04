import numpy as np
from .common import get_dtype
from .Bcast import Bcast

def Scatter(self, world, dtype=None, ndim=None, root=0):
    """ScatterV a numpy array to all ranks in an MPI communicator.

    Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. See the example for more details. Must be called collectively.

    Parameters
    ----------
    self : numpy.ndarray
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

    if world.rank == root:
        assert (self.size % world.size) == 0, ValueError("Cannot use Scatter for arrays whose size is not equally divisible by the number of MPI ranks\nUse Scatterv instead")

    if dtype is None:
        dtype = world.bcast(get_dtype(self, world, rank=root), root=root)

    if ndim is None:
        # Broadcast the number of dimensions
        ndim = Bcast(np.ndim(self), world, root=root, ndim=0, dtype='int64')

    if (ndim == 1):  # For a 1D Array
        shape = int(Bcast(np.size(self), world, root=root, ndim=0, dtype='int64') / world.size)
        this = np.empty(shape, dtype=dtype)
        world.Scatter([self, None], this, root=root)
        return this

    assert False, ValueError("Scatter ndim must equal 1")