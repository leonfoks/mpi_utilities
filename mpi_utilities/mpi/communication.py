"""
Contains communication based routines for general things
"""
import numpy as np
from .Gather import Gather

def get_ranks_using_bool(value, *, comm, root=None):
    """Given a boolean value on each rank, return the ranks where the value is True.
    """
    values = Gather(value, comm=comm, root=root)
    if values is not None:
        values = np.atleast_1d(np.squeeze(np.argwhere(values)))
    return values

def listen(comm, rank=0):
    """Listen for requests from arbitrary ranks
    """
    from mpi4py import MPI

    assert comm.rank == rank, ValueError(f"Do not call listen on ranks != {rank}")

    status = MPI.Status()
    dummy = comm.recv(source = MPI.ANY_SOURCE, tag = 999, status = status)
    requesting_rank = status.Get_source()
    return requesting_rank

def request(comm, rank=0):
    """Send a request for new work.
    """
    assert not comm.rank == rank, ValueError(f"Do not call request on comm.rank == {rank}")
    comm.send(1, dest=rank, tag=999)