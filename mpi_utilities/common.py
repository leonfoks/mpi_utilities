import sys
import numpy as np
import progressbar
import pickle

def print(*args, comm=None, rank=None, show_rank=False, **kwargs):
    """Print to the screen with a flush.

    Parameters
    ----------
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.
    comm : mpi4py.MPI.Comm
        MPI parallel communicator.
    rank : int
        The rank to print from, default is the head rank, 0.

    """
    msg = ' '.join(str(x) for x in args) + "\n"
    if comm is None:
        sys.stdout.write(msg)
    else:
        if rank is None:
            msg = f"rank={comm.rank}:" + msg
            sys.stdout.write(msg)
        else:
            if show_rank:
                msg = f"rank={comm.rank}:" + msg
            if (comm.rank == rank):
                sys.stdout.write(msg)
    sys.stdout.flush()

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

def prange(*args, comm, root=0, **kwargs):
        """Generate a loop range.

        Tracks progress on the master rank only if parallel.

        Parameters
        ----------
        value : int
            Size of the loop to generate
        """
        bar = range(*args, **kwargs)

        if comm.rank == root:
            Bar = progressbar.ProgressBar()
            bar = Bar(bar)
        return bar


