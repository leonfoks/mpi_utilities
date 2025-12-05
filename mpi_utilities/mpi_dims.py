from pprint import pprint
import numpy as np
from .chunking import load_balance, cartesian_chunks
from .mpi_dim import mpi_dim

class mpi_dims(dict):
    """Chunked out of memory metadata for dimensions
    """

    def __init__(self, *, comm, shape=None, scheme='single', ibuffer=None, **kwargs):
        """Create chunking metadata on each rank for a Dataset or DataArray with dims.

        Parameters
        ----------
        shape : int, tuple, xarray.Dataset, xarray.Dataarray
            Shape of something that needs chunking
        comm : MPI.COMM_WORLD
            MPI communicator
        ibuffer : dict
            Number of cells to the left and right on a dimension to buffer the distributed region
        name : str, required if shape is int or tuple
            The name of this thing
        dims : tuple, required if shape is int or tuple
            Dimension names of the array

        """

        self.scheme = scheme

        # I am already an xarray DataArray
        if not (np.isscalar(shape) or isinstance(shape, (tuple, list))):
            dims = kwargs.get('dims', shape.dims)
            name = kwargs.get('name', 'None')
            shape = [shape.sizes[dim] for dim in dims]

        else:
            name = kwargs.get('name', 'None')
            assert isinstance(name, str), ValueError("name should be a single string")

            if shape is None:
                assert 'coords' in kwargs, ValueError("At least a shape or coords must be given")
                shape = kwargs['coords'].mpi.global_shape

            if "coords" in kwargs:
                dims = {key: value.mpi.global_shape for key, value in kwargs['coords'].items()}

            elif "dims" in kwargs:
                dims = kwargs.pop('dims')
                if isinstance(dims, dict):
                    shape = list(dims.values())
                    dims = list(dims.keys())
            else:
                # if np.size(shape) > 1:
                #     assert False, ValueError("Must specify dims when shape has size > 1")
                dims = [name]

            dims = kwargs.get('dims', dims)
            shape = np.atleast_1d(shape)
            dtype = kwargs.get('dtype', [np.float64 for dim in dims])

        match scheme.lower():
            case "single":
                # Get the start index, chunk size, and grid shape for the MPI ranks
                starts, chunks, topo = load_balance(shape, comm.size)

                s = np.atleast_1d(starts[comm.rank])
                c = np.atleast_1d(chunks[comm.rank])

                indices = np.asarray([np.asarray([a, a+b]) for a, b in zip(s, c)])
                chunks = np.atleast_2d(chunks)

                for dim, shap, chunk, n_chunk in zip(dims, shape, chunks, topo):
                    self[dim] = mpi_dim(dim, shap, chunk, n_chunk)

                self.inner_indices = indices

            case "chunked":
                print('NADA')

            case "queue":
                assert 'increment' in kwargs, ValueError("For the queue scheme, specify an increment for each parallel axis")
                indices = cartesian_chunks(shape, **kwargs)

                for dim, shap, index in zip(dims, shape, indices):
                    self[dim] = mpi_dim(dim, shap, indices=index)

        if ibuffer is not None:
            self.ibuffer = ibuffer

    @property
    def summary(self):
        out = ""
        for k, v in self.items():
            out += f"{k}\n"
            out += v.summary
        return out

    @property
    def dims(self):
        return tuple(self.keys())

    @property
    def dtype(self):
        return {dim: self[dim].dtype for dim in self}

    @dtype.setter
    def dtype(self, values={}):
        for dim in self:
            if dim in values:
                self[dim].dtype = values[dim]

    @property
    def n_dims(self):
        return len(self)

    @property
    def global_shape(self):
        return {dim: self[dim].global_shape for dim in self}

    @property
    def ibuffer(self):
        return {dim: self[dim].ibuffer for dim in self}

    @ibuffer.setter
    def ibuffer(self, values):

        if isinstance(values, dict):
            for dim, val in values.items():
                self[dim].ibuffer = val
        else:
            for i, dim in enumerate(self):
                self[dim].ibuffer = values[i]

    @property
    def indirect(self):
        return [dim.indirect for dim in self]

    @property
    def inner_indices(self):
        return {dim: self[dim].inner_indices for dim in self}

    @inner_indices.setter
    def inner_indices(self, values):
        for i, dim in enumerate(self):
            self[dim].inner_indices = values[i]

    @property
    def inner_shape(self):
        return np.asarray([self[dim].inner_shape for dim in self], dtype=np.int64)

    @property
    def inner_slice(self):
        return [self[dim].inner_slice for dim in self]

    @property
    def inner_slice_xr(self):
        return {dim: self[dim].inner_slice for dim in self}

    @property
    def is_buffered(self):
        return {dim: self[dim].is_buffered for dim in self}

    @property
    def n_chunks(self):
        return np.asarray([self[dim].n_chunks for dim in self])

    @property
    def outer_indices(self):
        return {dim: self[dim].outer_indices for dim in self}

    # @outer_indices.setter
    # def outer_indices(self, values):
    #     """Indices into the 2D array of the outer buffered window

    #     Parameters
    #     ----------
    #     values : ints
    #         Length 4, ordered as [ymin, xmin, ymax, xmax]

    #     """
    #     for i, dim in enumerate(self):
    #         self[dim].outer_indices = values[i]

    @property
    def outer_shape(self):
        return np.squeeze(np.asarray([self[dim].outer_shape for dim in self], dtype=np.int64))

    @property
    def outer_slice(self):
        return [self[dim].outer_slice for dim in self]

    @property
    def outer_slice_xr(self):
        return {dim: self[dim].outer_slice for dim in self}

    @property
    def scheme(self):
        return {dim: self[dim].scheme for dim in self}

    @scheme.setter
    def scheme(self, value):
        for i, dim in enumerate(self):
            self[dim].scheme = value

    def squeeze(self, name=None):
        if len(self) == 1:
            return self[self.dims[0]]
        if name in self:
            return self[name]
        return self
