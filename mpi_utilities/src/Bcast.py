import numpy as np
from .common import get_dtype

def Bcast(self, comm, root=0, dtype=None, ndim=None, shape=None):
    """Broadcast a string or a numpy array

    Broadcast a string or a numpy array from a root rank to all ranks in an MPI communicator.
    Must be called collectively.
    In order to call this function collectively, the variable 'self' must be instantiated on every rank.
    See the example section for more details.

    Parameters
    ----------
    self : str or numpy.ndarray
        A string or numpy array to broadcast from root.
    comm : mpi4py.MPI.Comm
        MPI parallel communicator.
    root : int, optional
        The MPI rank to broadcast from. Default is 0.
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
    out : same type as self
        The broadcast object on every rank.

    Raises
    ------
    TypeError
        If self is a list, tell the user to use the specific Bcast_list function.
        While it has less code and seems like it might be faster, MPI actually pickles the list,
        broadcasts that binary stream, and unpickles on the other side. For a large number of lists,
        this can take a long time. This way, the user is made aware of the time benefits of using numpy arrays.

    Examples
    --------
    Given a numpy array instantiated on the head rank 0, in order to broadcast it,
    I must also instantiate a variable with the same name on all other ranks.

    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> import mpi4py_utilities as mpiu
    >>> comm = MPI.COMM_WORLD
    >>> if comm.rank == 0:
    >>>     x = np.arange(10)
    >>> # Instantiate on all other ranks before broadcasting
    >>> else:
    >>>     x=None
    >>> y = mpiu.Bcast(x, comm)
    >>>
    >>> # A string example
    >>> if (comm.rank == 0):
    >>>     s = 'some string'  # This may have been read in through an input file for production code
    >>> else:
    >>>     s = ''
    >>> s = mpiu.Bcast(s,comm)

    """
    # Broadcast the data type
    if dtype is None:
        dtype = comm.bcast(get_dtype(self, comm, rank=root), root=root)

    if dtype == 'str':
        return comm.bcast(self, root=root)

    if dtype == 'list':
        return comm.bcast(self, root=root)

    # Broadcast the number of dimensions
    if ndim is None:
        ndim = Bcast(np.ndim(self), comm, root=root, ndim=0, dtype='int64')

    if (ndim == 0):  # For a single number
        this = np.empty(1, dtype=dtype)  # Initialize on each worker
        if (comm.rank == root):
            this[0] = self  # Assign on the head
        comm.Bcast(this)  # Broadcast
        return this[0]

    if (ndim == 1):  # For a 1D array
        if shape is None:
            shape = Bcast(np.size(self), comm, root=root, ndim=0, dtype='int64')  # Broadcast the array size
        this = np.empty(shape, dtype=dtype)
        if (comm.rank == root):  # Assign on the root
            this[:] = self
        comm.Bcast(this, root=root)  # Broadcast
        return this

    if (ndim > 1):  # nD Array
        if shape is None:
            shape = Bcast(np.asarray(np.shape(self)), comm, root=root, shape=ndim, dtype='int64')  # Broadcast the shape
        this = np.empty(shape, dtype=dtype)
        if (comm.rank == root):  # Assign on the root
            this[:] = self
        comm.Bcast(this, root=root)  # Broadcast
        return this
