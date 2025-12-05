from pprint import pprint
import numpy as np
from .chunking import load_balance, cartesian_chunks

class mpi_dim(dict):
    """Define a chunked dimension with no large memory values

    The dimension is distributed amongst MPI ranks in a communicator
    """
    def __init__(self, dim, global_shape, chunks=None, n_chunks=None, indices=None, ibuffer=None, scheme='single'):
        self.dims = dim
        self.global_shape = global_shape
        self.chunks = chunks
        self.n_chunks = n_chunks

        if indices is not None:
            self.inner_indices = indices

        self.ibuffer = ibuffer
        self.scheme = scheme

    @property
    def summary(self):
        out = ""
        for k,v in self.items():
            out += f"  {k}:{v}\n"
        return out

    @property
    def chunks(self):
        return self.get('__chunks')

    @chunks.setter
    def chunks(self, values):
        self['__chunks'] = np.atleast_1d(values)

    @property
    def buffers(self):
        return self.get('__buffers')

    @buffers.setter
    def buffers(self, value):
        assert isinstance(value, dict), ValueError("buffers must have type dict")
        self['__buffers'] = value

    @property
    def dims(self):
        return self.get('__dims')

    @dims.setter
    def dims(self, value):
        self['__dims'] = value

    @property
    def dtype(self):
        return self.get('__dtype')

    @dtype.setter
    def dtype(self, value):
        self['__dtype'] = value

    @property
    def global_shape(self):
        return self.get('__global_shape')

    @global_shape.setter
    def global_shape(self, value):
        self['__global_shape'] = value

    @property
    def ibuffer(self):
        return self.get('__ibuffer', None if self.indirect else np.r_[0, 0])

    @ibuffer.setter
    def ibuffer(self, values=None):

        if self.indirect:
            if not values is None:
                assert 0 <= np.min(values), ValueError(f"outer_indices out of bounds {self.global_shape}")
                assert np.max(values) < self.global_shape, ValueError(f"outer_indices out of bounds {self.global_shape}")

                # assert that inner_indices and ibuffer are uniquely different
                # assert ~np.any(np.isin(values, self.inner_indices))
        else:

            # outer = np.zeros(2, dtype=np.int64)
            if values is None:
                values = np.r_[0, 0]
            if np.size(values) == 1:
                values = np.r_[values, values]

            # # Set
            # if not isinstance(self.inner_indices, zip):
            #     outer[0] = np.maximum(self.inner_indices[0] - values[0], 0)
            #     outer[1] = np.minimum(self.inner_indices[1] + values[1], self.global_shape)

            #     self.outer_indices = outer

        self['__ibuffer'] = values

    @property
    def indirect(self):
        return self.get('__indirect', False)

    @indirect.setter
    def indirect(self, value):
        assert isinstance(value, bool), TypeError("indirect must have type bool")
        self['__indirect'] = value

    @property
    def inner_indices(self):
        return self.get('__inner_indices')

    @inner_indices.setter
    def inner_indices(self, values):
        """Indices into the 2D array of the unbuffered window

        Parameters
        ----------
        values : ints
            Length 4, ordered as [ymin, xmin, ymax, xmax]

        """
        # assert self.is_dim, TypeError("Can only set inner_indices on dimensions")
        if isinstance(values, np.ndarray):
            assert (np.all(0 <= values) & np.all(values <= self.global_shape)), ValueError(f"inner_indices {values} must be ordered and in bounds {self.global_shape}")
        self['__inner_indices'] = values
        # self['__outer_indices'] = values

    @property
    def inner_shape(self):
        return self.inner_indices[1] - self.inner_indices[0]

    @property
    def inner_slice(self):
        """
        Returns
        -------
        slice
            Inner 2D slice of an unbuffered window
        """
        if self.indirect:
            return self.inner_indices
        else:
            return np.s_[self.inner_indices[0]:self.inner_indices[1]]

    @property
    def is_buffered(self):
        if self.ibuffer is None:
            return False
        else:
            return np.all(self.ibuffer > 0)

    @property
    def n_dims(self):
        return 1

    @property
    def n_chunks(self):
        return self['__n_chunks']

    @n_chunks.setter
    def n_chunks(self, value):
        self['__n_chunks'] = value

    @property
    def outer_indices(self):
        """Indices into the 2D array of the outer buffered window

        Ordered as [y0, x0, y1, x1]

        Returns
        -------
        ints
            Outer indices of a buffered window
        """
        return np.r_[np.maximum(self.inner_indices[0] - self.ibuffer[0], 0),
                     np.minimum(self.inner_indices[1] + self.ibuffer[1], self.global_shape)]
        # return self.get('__outer_indices')

    # @outer_indices.setter
    # def outer_indices(self, values):
    #     """Indices into the 2D array of the outer buffered window

    #     Parameters
    #     ----------
    #     values : ints
    #         Length 4, ordered as [ymin, xmin, ymax, xmax]

    #     """
    #     # assert self.is_dim, TypeError("Can only assign outer_indices on dimensions")

    #     if self.indirect:
    #         assert 0 <= np.min(values), ValueError(f"outer_indices out of bounds {self.global_shape}")
    #         assert np.max(values) < self.global_shape, ValueError(f"outer_indices out of bounds {self.global_shape}")

    #     else:
    #         assert np.size(values) == 2, ValueError("outer_indices must have length 2")
    #         assert values[1] > values[0], ValueError('xmax must be > xmin')

    #         assert 0 <= values[0] < self.global_shape, ValueError(f"outer_indices out of bounds {self.global_shape}")
    #         assert 0 < values[1] <= self.global_shape, ValueError(f"outer_indices out of bounds {self.global_shape}")

    #         assert np.all(values[0] <= self.inner_indices) and np.all(self.inner_indices <= values[1]), ValueError("outer_indices cannot be inside the inner_indices")

    #     self['__outer_indices'] = values

    @property
    def outer_shape(self):
        return np.int64(self.outer_indices[1] - self.outer_indices[0])

    @property
    def outer_slice(self):
        if self.indirect:
            return self.outer_indices
        else:
            return np.s_[self.outer_indices[0]:self.outer_indices[1]]

    @property
    def scheme(self):
        return self.get('__scheme')

    @scheme.setter
    def scheme(self, value):
        assert isinstance(value, str), ValueError("scheme must be a str")
        self['__scheme'] = value