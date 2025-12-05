import numpy as np
from mpi4py.MPI import Intracomm as mpi4py_Intracomm
from mpi4py.MPI import Cartcomm as mpi4py_Cartcomm
from ..mpi_dims import mpi_dims
from ..chunking import load_balance as mpiu_load_balance

class Intracomm(mpi4py_Intracomm):

    def __hash__(self):
        return id(self)

    def load_balance(self, shape, **kwargs):
        starts, chunks, rank_dims = mpiu_load_balance(shape, self.size, **kwargs)
        return starts, chunks, self.as_cartesian(dims=rank_dims)

    def as_cartesian(self, dims=None, reorder=True, **kwargs):
        assert np.prod(dims) == self.size, ValueError("Product of n_chunks must equal communicator size")
        return Cartcomm(self.Create_cart(dims=dims, reorder=reorder, **kwargs))

class Cartcomm(mpi4py_Cartcomm):

    def unravel_rank(self):
        return self.Get_coords(self.rank)

    def axis_index(self, axis):
        return self.unravel_rank()[axis]

    def split_along_axis(self, axis):
        keep_dims = [True for i in self.dims]
        keep_dims[axis] = False
        # return self.Split(color=self.axis_index(axis), key=self.rank)
        return self.Sub(keep_dims) # specific to cartesian communicators
