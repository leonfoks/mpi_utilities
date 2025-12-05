import numpy as np
from .prng import prng as mpi_prng
from ..mpi.Allgather import Allgather
from ..mpi.Bcast import Bcast

def choice(values, *, comm, k=None, rank=None, **kwargs):

    prng = kwargs.get('prng', None)
    if prng is None:
        prng = mpi_prng(comm=comm)

    # Obtain the ranks with non-empty arrays
    chunks = Allgather(values.size, comm=comm)
    ranks = np.atleast_1d(np.squeeze(np.argwhere(chunks > 0)))

    if rank is None:
        # randomly choose a rank
        rank = Bcast(prng.choice(ranks, size=1), root=0, comm=comm, dtype=np.int64)
    assert rank in ranks, ValueError("Chosen rank has no values to choose from")

    # Get the value from that rank and broadcast it
    val = None
    if comm.rank == rank:
        val = prng.choice(values, size=1) if k is None else values[k]

    return Bcast(val, root=rank, comm=comm, ndim=1, dtype=values.dtype).item()