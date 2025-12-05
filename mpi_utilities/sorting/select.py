from copy import copy
import numpy as np
from numba_stdlib import partition_3_way, argpartition_3_way
from ..mpi import Allreduce, Allgather, Bcast, communication, Gather
from ..random import prng as mpi_prng
from ..random import choice

def select(values, k, *, comm, **kwargs):
    """
    Parallel Quickselect implementation using MPI.
    Finds the k-th smallest element in a distributed array.
    """
    prng = kwargs.get('prng', None)
    if prng is None:
        kwargs['prng'] = mpi_prng(comm=comm)

    i0 = 0
    i1 = values.size

    kth = copy(k)

    while True:
        # A random pivot from a rank is broadcast
        pivot = choice(values[i0:i1], comm=comm, k=0, **kwargs)

        p0, p1 = partition_3_way(values, i0, i1, pivot)

        # Size of the first two partitions
        local_sm = p0 - i0
        local_eq = p1 - p0

        global_sm = Allreduce(local_sm, op='sum', comm=comm)
        global_eq = Allreduce(local_eq, op='sum', comm=comm)
        global_leq = global_sm + global_eq

        # Check which partition contains the k-th element
        if kth < global_sm:
            # Move to the "smaller" partition
            i1 = p0

        elif kth < global_leq:
            # Get the ranks that contains the pivot. There could be multiple where the kth value is repeated.
            # rank = communication.get_ranks_using_bool(local_eq > 0, comm=comm)
            # # Broadcast the pivot from that rank
            # return Bcast(pivot, root=rank[0], comm=comm, ndim=1, shape=1, dtype=dtype)
            # We shouldnt need a global comm here, the pivot was bcast already.
            return pivot
        else:
            # Move to the "larger" partition and adjust k
            kth -= global_leq
            i0 = p1

def arg_select(values, index, k, *, comm, **kwargs):
    """
    Parallel Quickselect implementation using MPI.
    Finds the k-th smallest element in a distributed array.
    """
    prng = kwargs.get('prng', None)
    if prng is None:
        kwargs['prng'] = mpi_prng(comm=comm)

    i0 = 0
    i1 = values.size

    kth = copy(k)

    rank = comm.rank

    while True:
        # A random pivot from a rank is broadcast
        pivot = choice(values[index[i0:i1]], comm=comm, k=0, **kwargs)

        p0, p1 = argpartition_3_way(values, index, i0, i1, pivot)

        # Size of the first two partitions
        local_sm = p0 - i0
        local_eq = p1 - p0

        global_sm = Allreduce(local_sm, op='sum', comm=comm)
        global_eq = Allreduce(local_eq, op='sum', comm=comm)
        global_leq = global_sm + global_eq

        # Check which partition contains the k-th element
        if kth < global_sm:
            # Move to the "smaller" partition
            i1 = p0

        elif kth < global_leq:
            # Get the cumulative number of values that are less than the pivot
            rank = communication.get_ranks_using_bool(p1-p0 > 0, comm=comm)[0]

            # Broadcast this ranks local index of the pivot
            local_index = Bcast(index[p0], root=rank, comm=comm, dtype=int, ndim=0, shape=1)

            return rank, local_index
        else:
            # Move to the "larger" partition and adjust k
            kth -= global_leq
            i0 = p1