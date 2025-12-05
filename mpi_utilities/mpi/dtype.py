def mpiu_dtype(self):
    out = str(self.__class__.__name__)
    if 'ndarray' in out:
        out = str(self.dtype)
    return out

def get_dtype(self, comm=None, rank=0):
    out = None
    if comm is not None:
        if comm.rank == rank:
            out = mpiu_dtype(self)
    return out