from mpi4py.MPI import Cartcomm
import numpy as np
from ..mpi import Reduce

def dot(a, b, *, comm, out=None, **kwargs):
    """Compute the dot product of distributed arrays.

    Parameters
    ----------
    a : ndarray
        Local array of values.
    b : ndarray
        Local array of values.
    comm : MPI.Comm
        MPI communicator.
    out : ndarray, optional
        Output array to store the result.

    Returns
    -------
    dot : float
        The dot product of the distributed arrays.
    """

    assert isinstance(comm, Cartcomm), TypeError("comm must be a Cartcomm instance")

    out = np.dot(a, b, out=out)

    split = comm.split_along_axis(axis=1)

    return Reduce(out, op='sum', comm=split, inplace=True, **kwargs)
