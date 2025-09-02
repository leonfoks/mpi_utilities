import numpy as np
from .common import get_dtype, print, request


def Irecv(source, comm, dtype=None, ndim=None, shape=None, listen_request=False, verbose=False, **kwargs):
    """Irecv wrapper.

    Automatically determines data type and shape. Must be accompanied by Isend on the source rank.

    Parameters
    ----------
    source : int
        Rank to receive to
    comm : mpi4py.MPI.COMM_WORLD
        MPI communicator
    dtype : dtype, optional
        Pre-determined data type if known.
        Defaults to None.
    ndim : int, optional
        Number of dimension if known.
        Defaults to None.
    shape : ints, optional
        values shape if known.
        Defaults to None.

    Returns
    -------
    out : scalar or array_like
        returned type depends on what was sent.
    """
    if listen_request:
        request(comm=comm, rank=source)

    if dtype is None:
        dtype = comm.irecv(source=source).wait()
    if verbose:
        print(f"{dtype=}")

    assert not dtype == 'list', TypeError("Cannot Isend/Irecv a list")

    if dtype == 'str':
        this = comm.irecv(source=source).wait()
        return this

    if ndim is None:
        ndim = Irecv(source, comm, ndim=0, dtype=np.int64)
    if verbose:
        print(f"{ndim=}")

    if (ndim == 0):  # For a single number
        this = np.empty(1, dtype=dtype)  # Initialize on each worker
        comm.Irecv(this, source=source).Wait()
        this = this[0]

    elif (ndim == 1): # For a 1D array
        if shape is None:
            shape = Irecv(source=source, comm=comm, ndim=0, dtype=np.int64)
        if verbose:
            print(f"{shape=}")
        this = np.empty(shape, dtype=dtype)
        comm.Irecv(this, source=source).Wait()

    elif (ndim > 1): # Nd Array
        if shape is None:
            shape = Irecv(source=source, comm=comm, shape=ndim, dtype=np.int64)
        if verbose:
            print(f"{shape=}")
        this = np.empty(shape, dtype=dtype)
        comm.Irecv(this, source=source).Wait()

    return this
