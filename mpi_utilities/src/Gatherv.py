import numpy as np
from .common import mpiu_dtype, load_balance
from .Bcast import Bcast
from .Gather import Gather
from .Allgather import Allgather

def Gatherv(self, world, starts=None, chunks=None, root=0):
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
    # if dtype is None:
    dtype = mpiu_dtype(self)
    # world.bcast(get_dtype(self, world, rank=root), root=root)

    # if ndim is None:
        # Broadcast the number of dimensions
        # ndim = Bcast(np.ndim(self), world, root=root, ndim=0, dtype='int64')
    ndim = np.ndim(self)

    if ndim == 0:
        return Gather(self, world, root=root)

    this = None
    if ndim == 1:
        if chunks is None:
            chunks = Allgather(np.size(self), world)

        if starts is None:
            starts = np.hstack([0, np.cumsum(chunks[:-1])])

        if world.rank == root:
            this = np.empty(np.sum(chunks), dtype=dtype)

        world.Gatherv(self, [this, chunks, starts, None], root=root)
        return this