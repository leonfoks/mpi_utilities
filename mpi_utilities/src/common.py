import sys
import numpy as np
import progressbar
import pickle
from numpy.random import Generator, PCG64DXSM
from .numba_methods import best_fit_chunks

def prng(generator=PCG64DXSM, seed=None, jump=None, world=None):
    """Generate an independent prng.

    Returns
    -------
    seed : int or file, optional
        The seed of the bit generator.
    jump : int, optional
        Jump the bit generator by this amount
    world : mpi4py.MPI.COMM_WORLD, optional
        MPI communicator, will jump each bit generator by world.rank

    """
    # Default to single core, else grab the mpi rank.
    rank = 0
    if world is not None:
        rank = world.rank

    if rank == 0:
        if seed is not None: # Seed is a file.
            if isinstance(seed, str):
                with open(seed, 'rb') as f:
                    seed = pickle.load(f)
            assert isinstance(seed, int), TypeError("Seed {} must have type python int (not numpy)".format(seed))

        else: # No seed, generate one
            bit_generator = generator()
            seed = bit_generator.seed_seq.entropy
            with open('seed.pkl', 'wb') as f:
                pickle.dump(seed, f)
            print('Seed: {}'.format(seed), flush=True)

    if world is not None:
        # Broadcast the seed to all ranks.
        seed = world.bcast(seed, root=0)

    bit_generator = generator(seed = seed)

    if world is not None:
        jump = world.rank

    if jump is not None:
        bit_generator = bit_generator.jumped(jump)

    return Generator(bit_generator)

def get_dtype(self, world=None, rank=0):
    out = None
    if world is not None:
        if world.rank == rank:
            out = mpiu_dtype(self)
    return out

def mpiu_dtype(self):
    out = str(self.__class__.__name__)
    if 'ndarray' in out:
        out = str(self.dtype)
    return out

def mpiu_time(world=None):
    import time
    if world is None:
        return time.time()
    else:
        from mpi4py.MPI import Wtime
        return Wtime()

def print(*args, world=None, rank=None, **kwargs):
    """Print to the screen with a flush.

    Parameters
    ----------
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    rank : int
        The rank to print from, default is the head rank, 0.

    """
    msg = ' '.join(str(x) for x in args) + "\n"
    if world is None:
        sys.stdout.write(msg)
    else:
        msg = f"{world.rank=}:" + msg
        if rank is None:
            sys.stdout.write(msg)
        else:
            if (world.rank == rank):
                sys.stdout.write(msg)
    sys.stdout.flush()

def load_balance(shape, n_chunks, flatten=True):

    if (np.ndim(shape) == 0) or ((np.ndim(shape) == 1) and (np.size(shape) == 1)):
        s, c = load_balance_1d(shape, n_chunks)
        return s, c, s+c

    bf = best_fit_chunks(shape, n_chunks)

    s1d = []; c1d = []
    for i in range(np.size(shape)):
        s, c = load_balance_1d(shape[i], bf[i])
        s1d.append(s); c1d.append(c)

    # Use itertools.product to get all combinations
    # The '*' unpacks the list of arrays as separate arguments to product
    starts = np.stack(np.meshgrid(*s1d, indexing='ij'), axis=-1).reshape(-1, len(s1d))
    chunks = np.stack(np.meshgrid(*c1d, indexing='ij'), axis=-1).reshape(-1, len(c1d))

    if  not flatten:
        starts = starts.reshape(*bf, np.size(shape))
        chunks = chunks.reshape(*bf, np.size(shape))

    return starts, chunks, starts + chunks

def load_balance_1d(N, n_chunks):
    """Splits the length of an array into a number of chunks. Load balances the chunks in a shrinking arrays fashion.

    Given length, N, split N up into n_chunks and return the starting index and size of each chunk.
    After being split equally among the chunks, the remainder is distributed so that chunks 0:remainder
    get +1 in size. e.g. N=10, n_chunks=3 would return starts=[0,4,7] chunks=[4,3,3]

    Parameters
    ----------
    N : int
        A size to split into chunks.
    n_chunks : int
        The number of chunks to split N into. Usually the number of ranks, world.size.

    Returns
    -------
    starts : ndarray of ints
        The starting indices of each chunk.
    chunks : ndarray of ints
        The size of each chunk.

    """
    chunks = np.full(n_chunks, fill_value=N/n_chunks, dtype=np.int64)
    mod = np.int64(N % n_chunks)
    chunks[:mod] += 1
    starts = np.cumsum(chunks) - chunks[0]
    if (mod > 0):
        starts[mod:] += 1
    return starts, chunks

def channels(shape):

    match np.ndim(shape):
        case 0:
            channels = np.tile(np.asarray([0, 1]), reps=np.int32(0.5 * shape))
        case 1:
            match np.size(shape):
                case 1:
                    channels = np.tile(np.asarray([0, 1]), reps=np.int32(0.5 * shape))
                case 2:
                    print('stuff')
                case 3:
                    print('stuff')

def prange(*args, world, root=0, **kwargs):
        """Generate a loop range.

        Tracks progress on the master rank only if parallel.

        Parameters
        ----------
        value : int
            Size of the loop to generate
        """
        bar = range(*args, **kwargs)

        if world.rank == root:
            Bar = progressbar.ProgressBar()
            bar = Bar(bar)
        return bar


def listen(world, rank=0):
    """Listen for requests from arbitrary ranks
    """
    from mpi4py import MPI

    assert world.rank == rank, ValueError(f"Do not call listen on ranks != {rank}")

    status = MPI.Status()
    dummy = world.recv(source = MPI.ANY_SOURCE, tag = 11, status = status)
    requesting_rank = status.Get_source()
    return requesting_rank

def request(world, rank=0):
    """Send a request for new work.
    """
    assert not world.rank == rank, ValueError(f"Do not call request on world.rank == {rank}")
    world.send(1, dest=rank, tag=11)


