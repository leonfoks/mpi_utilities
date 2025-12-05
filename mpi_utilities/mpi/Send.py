import numpy as np
from .dtype import get_dtype
from .communication import listen

def Send(values, dest, comm, dtype=None, ndim=None, shape=None, listen_request=False, verbose=False, **kwargs):
    """Isend wrapper.

    Automatically determines data type and shape. Must be accompanied by Irecv on the dest rank.

    Parameters
    ----------
    dest : int
        Rank to send to
    comm : mpi4py.MPI.COMM_WORLD
        MPI communicator
    dtype : dtype, optional
        Pre-determined data type if known. Faster
        Defaults to None.
    ndim : int, optional
        Number of dimension if known. Faster
        Defaults to None.
    shape : ints, optional
        values shape if known. Faster
        Defaults to None.
    """

    if listen_request:
        dest = listen(comm=comm)

    # Send the data type
    if dtype is None:
        dtype = get_dtype(values, comm, comm.rank)
        comm.send(dtype, dest=dest)

    assert (not dtype == 'list'), TypeError("Cannot Send/Recv a list")

    if dtype == 'str':
        return comm.send(values, dest=dest)

    # Broadcast the number of dimensions
    if ndim is None:
        ndim = np.ndim(values)
        Send(ndim, dest=dest, comm=comm, ndim=0, dtype=np.int64)

    if (ndim == 0):  # For a single number
        this = np.full(1, values, dtype=dtype)  # Initialize on each worker
        comm.Send(this, dest=dest, **kwargs)

    elif (ndim == 1):  # For a 1D array
        if shape is None:
            Send(np.size(values), dest=dest, comm=comm, ndim=0, dtype=np.int64)
        comm.Send(values, dest=dest, **kwargs)

    elif (ndim > 1):  # nD Array
        if shape is None:
            Send(np.asarray(values.shape, dtype=np.int64), dest=dest, comm=comm, shape=ndim, dtype=np.int64)
        comm.Send(values, dest=dest, **kwargs)

    return dest