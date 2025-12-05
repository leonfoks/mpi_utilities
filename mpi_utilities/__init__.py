""" Module containing custom MPI functions """
import sys
from os import getpid
import pickle
import numpy as np
from .common import print
from .mpi_dims import mpi_dims

from .mpi import Intracomm, Cartcomm
from .mpi import Allgather
from .mpi import Allgatherv
from .mpi import Allreduce
from .mpi import Bcast
from .mpi import Gather
from .mpi import Gatherv
from .mpi import Isend
from .mpi import Irecv
from .mpi import Recv
from .mpi import Reduce
from .mpi import Scatter
from .mpi import Scatterv
from .mpi import Send

from . import random, maths, linalg, sorting, array, mpi, chunking

def Stopwatch(**kwargs):
    from mpi4py import MPI
    from stopywatch import Stopwatch as Stopywatch
    return Stopywatch(timer=MPI.Wtime, **kwargs)

#from ...base.Error import Error as Err

def banner(comm, *args, root=0, **kwargs):
    """Prints a String with Separators above and below

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI parallel communicator.
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.
    rank : int
        The rank to print from, default is the head rank, 0.

    """
    comm.barrier()

    if comm.rank == root:
        msg = "="*78
        print(msg, flush=True)
        print(*args, flush=True)
        print(msg, flush=True)
    comm.barrier()


def ordered_print(comm, values, title=None):
    """Prints numbers from each rank in order of rank

    This routine will print an item from each rank in order of rank.
    This routine is SLOW due to lots of communication, but is useful for illustration purposes, or debugging.
    Do not use this in production code!  The title is used in a banner

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI parallel communicator.
    values : array_like
        Variable to print, must exist on every rank in the communicator.
    title : str, optional
        Creates a banner to separate output with a clear indication of what is being written.

    """
    if (comm.rank > 0):
        comm.send(values, dest=0, tag=14)
    else:
        banner(comm, title)
        print('Rank 0 {}'.format(values))
        for i in range(1, comm.size):
            tmp = comm.recv(source=i, tag=14)
            print("Rank {} {}".format(i, tmp))


def hello_world(*, comm):
    """Print hello from every rank in an MPI communicator

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI parallel communicator.

    """
    from mpi4py import MPI
    print(f"Hello from {comm.rank}/{comm.size} on {MPI.Get_processor_name()}", flush=True)


# def Scatterv_list(self, starts, chunks, comm, root=0):
#     """Scatterv a list by pickling, sending, receiving, and unpickling.  This is slower than using numpy arrays and uppercase (Scatterv) mpi4py routines. Must be called collectively.

#     Parameters
#     ----------
#     self : list
#         A list to scatterv.
#     starts : array of ints
#         1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
#     chunks : array of ints
#         1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
#     comm : mpi4py.MPI.Comm
#         MPI parallel communicator.
#     root : int, optional
#         The MPI rank to broadcast from. Default is 0.

#     Returns
#     -------
#     out : list
#         A chunk of self on each MPI rank with size chunk[comm.rank].

#     """
#     for i in range(comm.size):
#         if (i != root):
#             if (comm.rank == root):
#                 this = self[starts[i]:starts[i] + chunks[i]]
#                 comm.send(this, dest=i)
#             if (comm.rank == i):
#                 this = comm.recv(source=root)
#                 return this
#     if (comm.rank == root):
#         return self[:chunks[root]]


# def Scatterv_numpy(self, starts, chunks, dtype, comm, axis=0, root=0):
#     """ScatterV a numpy array to all ranks in an MPI communicator.

#     Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. See the example for more details. Must be called collectively.

#     Parameters
#     ----------
#     self : numpy.ndarray
#         A numpy array to broadcast from root.
#     starts : array of ints
#         1D array of ints with size equal to the number of MPI ranks.
#         Each element gives the starting index for a chunk to be sent to that core.
#         e.g. starts[0] is the starting index for rank = 0.
#         Must exist on all ranks
#     chunks : array of ints
#         1D array of ints with size equal to the number of MPI ranks.
#         Each element gives the size of a chunk to be sent to that core.
#         e.g. chunks[0] is the chunk size for rank = 0.
#         Must exist on all ranks
#     dtype : type
#         The type of the numpy array being scattered. Must exist on all ranks.
#     comm : mpi4py.MPI.Comm
#         MPI parallel communicator.
#     axis : int, optional
#         Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
#     root : int, optional
#         The MPI rank to broadcast from. Default is 0.

#     Returns
#     -------
#     out : numpy.ndarray
#         A chunk of self on each MPI rank with size chunk[comm.rank].

#     """
#     # Broadcast the number of dimensions
#     ndim = Bcast(np.ndim(self), comm, root=root, ndim=0, dtype='int64')
#     if (ndim == 1):  # For a 1D Array
#         this = np.empty(chunks[comm.rank], dtype=dtype)
#         comm.Scatterv([self, chunks, starts, None], this[:], root=root)
#         return this

#     # For a 2D Array
#     # MPI cannot send and receive arrays of more than one dimension.
#     # Therefore higher dimensional arrays must be unpacked to 1D, and then repacked on the other side.
#     if (ndim == 2):
#         s = Bcast(np.size(self, 1 - axis), comm, root=root, ndim=0, dtype='int64')
#         tmpChunks = chunks * s
#         tmpStarts = starts * s
#         self_unpk = None
#         if (comm.rank == root):
#             if (axis == 0):
#                 self_unpk = np.reshape(self, np.size(self))
#             else:
#                 self_unpk = np.reshape(self.T, np.size(self))
#         this_unpk = np.empty(tmpChunks[comm.rank], dtype=dtype)
#         comm.Scatterv([self_unpk, tmpChunks, tmpStarts, None], this_unpk, root=root)
#         this = np.reshape(this_unpk, [chunks[comm.rank], s])

#         return this.T if axis == 1 else this