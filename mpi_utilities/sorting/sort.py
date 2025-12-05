"""
https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Nicolas-Barrios-Fall-2021.pdf
"""

import numpy as np
from ..mpi.Bcast import Bcast
from ..mpi.Gather import Gather

def sort(values, *, comm, **kwargs):
    """
    Sorts a distributed array using a sample sort algorithm with mpi4py.
    """
    comm_size = comm.size
    comm_rank = comm.rank

    # Locally sort the values
    values.sort()

    # Take a sample of 100 values on each rank
    num_samples = min(100, len(values))

    # Take those samples spaced equally across the array
    samples = values[::len(values)//num_samples][:num_samples].copy()

    # Gather the samples on a single core to choose a pivot
    all_samples = Gather(samples, root=0, comm=comm)

    pivots = None
    if comm_rank == 0:
        # Flatten the list of sample arrays
        all_samples.sort()
        # Select comm.size - 1 global pivots from the sorted samples
        pivots = all_samples[::all_samples.size // comm.size][1:]

    pivots = Bcast(pivots, root=0, comm=comm)

    # Loop over sampling to find globally optimal pivots.

    # Split the local array using the global pivots
    local_bins = np.searchsorted(values, pivots)

    # Distribute data from each rank to other ranks given the bins
    split_boundaries = np.r_[0, local_bins, values.size]
    split_chunks = np.diff(split_boundaries)
    split_starts = split_boundaries[:-1]

    # Each rank tells every other rank how much data it will send to every other rank
    recv_counts = np.empty(comm_size, dtype=np.int64)
    comm.Alltoall(split_chunks, recv_counts)

    # Now we can compute the displacements for the receive buffer
    recv_displacements = np.r_[0, np.cumsum(recv_counts[:-1])]

    # Create the receiving memory buffer
    recv_buffer = np.empty(np.sum(recv_counts), dtype=values.dtype)

    # Perform the variable-size data exchange
    # The MPI_Alltoallv call takes the buffer, counts, displacements, and datatype for both send and receive
    comm.Alltoallv([values, split_chunks, split_starts, None],
                   [recv_buffer, recv_counts, recv_displacements, None])

    # Do a final local sort
    recv_buffer.sort()

    return recv_buffer