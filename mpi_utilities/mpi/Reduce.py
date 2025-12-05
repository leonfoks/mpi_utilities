import numpy as np
from mpi4py import MPI
from .dtype import mpiu_dtype
from .operator import operator
from .Allreduce import Allreduce

def Reduce(self, op, *, comm, inplace=False, root=None):

    if root is None:
        return Allreduce(self, op=op, comm=comm, inplace=inplace)

    op = operator(op)

    ndim = np.ndim(self)
    dtype = mpiu_dtype(self)

    if ndim == 0:
        self = np.atleast_1d(self)

    reduced = None
    if comm.rank == root:
        if inplace:
            reduced = self
            self = MPI.IN_PLACE

    comm.Reduce(self, reduced, op=op, root=root)

    if (ndim == 0) and (comm.rank == root):
        reduced = reduced.item()

    return reduced