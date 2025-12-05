import numpy as np
from .var import var

def std(values, *, comm, **kwargs):
    """
    Compute the standard deviation of distributed values.

    Parameters
    ----------
    values : np.ndarray
        Local array of values.
    comm : MPI.Comm, optional
        The MPI communicator. If None, MPI.COMM_WORLD is used.
    **kwargs
        Additional keyword arguments passed to `variance`.

    Returns
    -------
    float
        The standard deviation of the distributed values.
    """
    out = var(values, comm=comm, **kwargs)
    if out is not None:
        return np.sqrt(out)
    return None