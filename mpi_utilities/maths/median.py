from .quantile import quantile

def median(values, *, comm, **kwargs):
    """Compute the median of distributed values.

    Parameters
    ----------
    values : ndarray
        Local array of values.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    median : float
        The median of the distributed values.
    """
    return quantile(values, 0.5, comm=comm, **kwargs)