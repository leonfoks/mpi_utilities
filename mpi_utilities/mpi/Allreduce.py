from mpi4py import MPI
import numpy as np
from .dtype import mpiu_dtype
from .operator import operator

def Allreduce(self, op, *, comm, inplace=False):

    op = operator(op)

    ndim = np.ndim(self)
    dtype = mpiu_dtype(self)

    if ndim == 0:
        self = np.atleast_1d(self)

    if inplace:
        reduced = self
        self = MPI.IN_PLACE
    else:
        reduced = np.empty_like(self)

    comm.Allreduce(self, reduced, op=op)

    if ndim == 0:
        reduced = reduced.item()

    return reduced