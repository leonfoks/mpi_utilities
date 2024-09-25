""" Module containing custom MPI functions """
import sys
from os import getpid
import pickle
import numpy as np
from numpy.random import Generator, PCG64DXSM
from .src.common import print, load_balance
from .src.Bcast import Bcast
from .src.Send import Send
from .src.Recv import Recv
from .src.Isend import Isend
from .src.Irecv import Irecv
from .src.Scatter import Scatter
from .src.Scatterv import Scatterv
from .src.Gatherv import Gatherv
from .src.Gather import Gather
from .src.Allgather import Allgather
from .src.Allgatherv import Allgatherv


#from ...base.Error import Error as Err

def banner(world, aStr=None, end='\n', rank=0):
    """Prints a String with Separators above and below

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.
    rank : int
        The rank to print from, default is the head rank, 0.

    """
    if (aStr is None):
        return
    msg = "="*78
    msg += end + aStr + end + aStr
    msg += "="*78
    print_mpi(world, msg, end=end, rank=rank)

def ordered_print(world, values, title=None):
    """Prints numbers from each rank in order of rank

    This routine will print an item from each rank in order of rank.
    This routine is SLOW due to lots of communication, but is useful for illustration purposes, or debugging.
    Do not use this in production code!  The title is used in a banner

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    values : array_like
        Variable to print, must exist on every rank in the communicator.
    title : str, optional
        Creates a banner to separate output with a clear indication of what is being written.

    """
    if (world.rank > 0):
        world.send(this, dest=0, tag=14)
    else:
        banner(world, title)
        print('Rank 0 {}'.format(this))
        for i in range(1, world.size):
            tmp = world.recv(source=i, tag=14)
            print("Rank {} {}".format(i, tmp))


def hello_world(world):
    """Print hello from every rank in an MPI communicator

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    """
    from mpi4py import MPI
    print(f'Hello from {world.rank}/{world.size} on {MPI.Get_processor_name()}', flush=True)

# def Bcast_list(self, world, root=0):
#     """Broadcast a list by pickling, sending, and unpickling.
#     This is slower than using numpy arrays and uppercase (Bcast) mpi4py routines.
#     Must be called collectively.

#     Parameters
#     ----------
#     self : list
#         A list to broadcast.
#     world : mpi4py.MPI.Comm
#         MPI parallel communicator.
#     root : int, optional
#         The MPI rank to broadcast from. Default is 0.

#     Returns
#     -------
#     out : list
#         The broadcast list on every MPI rank.

#     """
#     this = world.bcast(self, root=root)
#     return this


# def Scatterv_list(self, starts, chunks, world, root=0):
#     """Scatterv a list by pickling, sending, receiving, and unpickling.  This is slower than using numpy arrays and uppercase (Scatterv) mpi4py routines. Must be called collectively.

#     Parameters
#     ----------
#     self : list
#         A list to scatterv.
#     starts : array of ints
#         1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
#     chunks : array of ints
#         1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
#     world : mpi4py.MPI.Comm
#         MPI parallel communicator.
#     root : int, optional
#         The MPI rank to broadcast from. Default is 0.

#     Returns
#     -------
#     out : list
#         A chunk of self on each MPI rank with size chunk[world.rank].

#     """
#     for i in range(world.size):
#         if (i != root):
#             if (world.rank == root):
#                 this = self[starts[i]:starts[i] + chunks[i]]
#                 world.send(this, dest=i)
#             if (world.rank == i):
#                 this = world.recv(source=root)
#                 return this
#     if (world.rank == root):
#         return self[:chunks[root]]


# def Scatterv_numpy(self, starts, chunks, dtype, world, axis=0, root=0):
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
#     world : mpi4py.MPI.Comm
#         MPI parallel communicator.
#     axis : int, optional
#         Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
#     root : int, optional
#         The MPI rank to broadcast from. Default is 0.

#     Returns
#     -------
#     out : numpy.ndarray
#         A chunk of self on each MPI rank with size chunk[world.rank].

#     """
#     # Broadcast the number of dimensions
#     ndim = Bcast(np.ndim(self), world, root=root, ndim=0, dtype='int64')
#     if (ndim == 1):  # For a 1D Array
#         this = np.empty(chunks[world.rank], dtype=dtype)
#         world.Scatterv([self, chunks, starts, None], this[:], root=root)
#         return this

#     # For a 2D Array
#     # MPI cannot send and receive arrays of more than one dimension.
#     # Therefore higher dimensional arrays must be unpacked to 1D, and then repacked on the other side.
#     if (ndim == 2):
#         s = Bcast(np.size(self, 1 - axis), world, root=root, ndim=0, dtype='int64')
#         tmpChunks = chunks * s
#         tmpStarts = starts * s
#         self_unpk = None
#         if (world.rank == root):
#             if (axis == 0):
#                 self_unpk = np.reshape(self, np.size(self))
#             else:
#                 self_unpk = np.reshape(self.T, np.size(self))
#         this_unpk = np.empty(tmpChunks[world.rank], dtype=dtype)
#         world.Scatterv([self_unpk, tmpChunks, tmpStarts, None], this_unpk, root=root)
#         this = np.reshape(this_unpk, [chunks[world.rank], s])

#         return this.T if axis == 1 else this

# def get_prng(generator=PCG64DXSM, seed=None, jump=None, world=None):
#     """Generate an independent prng.

#     Returns
#     -------
#     seed : int or file, optional
#         The seed of the bit generator.
#     jump : int, optional
#         Jump the bit generator by this amount
#     world : mpi4py.MPI.COMM_WORLD, optional
#         MPI communicator, will jump each bit generator by world.rank

#     """
#     # Default to single core, else grab the mpi rank.
#     rank = 0
#     if world is not None:
#         rank = world.rank

#     if rank == 0:
#         if seed is not None: # Seed is a file.
#             if isinstance(seed, str):
#                 with open(seed, 'rb') as f:
#                     seed = pickle.load(f)
#             assert isinstance(seed, int), TypeError("Seed {} must have type python int (not numpy)".format(seed))

#         else: # No seed, generate one
#             bit_generator = generator()
#             seed = bit_generator.seed_seq.entropy
#             with open('seed.pkl', 'wb') as f:
#                 pickle.dump(seed, f)
#             print('Seed: {}'.format(seed), flush=True)

#     if world is not None:
#         # Broadcast the seed to all ranks.
#         seed = world.bcast(seed, root=0)

#     bit_generator = generator(seed = seed)

#     if world is not None:
#         jump = world.rank

#     if jump is not None:
#         bit_generator = bit_generator.jumped(jump)

#     return Generator(bit_generator)