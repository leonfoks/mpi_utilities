import numpy as np
from .common import mpiu_dtype
from .Bcast import Bcast

def _get_mpi_operator(this):
    """Get the corresponding MPI operator

    """

    from mpi4py import MPI

    if this == 'max':
        return MPI.MAX
    elif this == 'min':
        return MPI.MIN
    elif this == 'sum':
        return MPI.SUM
    elif this == 'prod':
        return MPI.PROD
    elif this == 'land':
        return MPI.LAND
    elif this == 'lor':
        return MPI.LOR
    elif this == 'band':
        return MPI.BAND
    elif this == 'bor':
        return MPI.BOR
    elif this == 'maxloc':
        return MPI.MAXLOC
    elif this == 'minloc':
        return MPI.MINLOC

    assert False, ValueError(f"this {this} must be from (max, min, sum, prod, land, lor, band, bor, maxloc, minloc)")

def Reduce(self, operator, world, root=0):

    operator = _get_mpi_operator(operator)

    ndim = np.ndim(self)
    dtype = mpiu_dtype(self)

    if ndim == 0:
        self = np.atleast_1d(self)

    reduced = np.zeros_like(self)
    world.Reduce(self, reduced, op=operator, root=root)

    return np.asarray(reduced, dtype=dtype)