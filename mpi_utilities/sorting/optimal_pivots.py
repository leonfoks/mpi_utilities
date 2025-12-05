import numpy as np
from ..chunking import starts_from_chunks
from ..mpi.Bcast import Bcast
from ..mpi.Gather import Gather
from ..mpi.Gatherv import Gatherv
from ..mpi.Reduce import Reduce
from ..common import print

def optimal_global_pivots(values, *, comm, **kwargs):
    """Compute global pivots such that the memory on each core is unchanged.
    """

    # Locally sort the values
    values.sort()
    print(f"{values=}", comm=comm)

    min_val = Reduce(values.min(), op='min', comm=comm)
    max_val = Reduce(values.max(), op='max', comm=comm)

    print(f"{min_val=} {max_val=}\n", comm=comm, rank=0)

    comm.Barrier()

    rank = comm.rank
    size = comm.size

    chunks = Gather(values.size, comm=comm, root=0)

    # Take a sample of 100 values on each rank
    n_samples = min(100, len(values))

    # Take those samples spaced equally across the array
    # We need to copy here so that the samples are contiguous in memory
    samples = values[::len(values)//n_samples].copy()

    sample_chunks = Gather(n_samples, comm=comm)
    sample_starts = starts_from_chunks(sample_chunks)

    # Gather the samples on a single core to choose a pivot
    all_samples = Gatherv(samples, starts=sample_starts, chunks=sample_chunks, root=0, comm=comm)

    pivots = None
    if rank == 0:
        # Sort the samples
        all_samples.sort()
        # Select comm.size - 1 global pivots from the sorted samples
        pivots = all_samples[::all_samples.size // comm.size][1:]

    pivots = Bcast(pivots, root=0, comm=comm)

    print(f"{pivots=}", comm=comm, rank=0)

    # Loop over sampling to find globally optimal pivots.

    for i in range(1):
        print(f"ITERATION {i=}", comm=comm, rank=0)
        # Split the local array using the global pivots
        local_bins = np.minimum(np.searchsorted(values, pivots), values.size-1)
        print(f"{local_bins=} {values.size=}")
        split_chunks = np.diff(np.r_[0, local_bins])
        print(f"{split_chunks=}")

        test_chunks = Reduce(split_chunks, op='sum', comm=comm, root=0)

        new_pivots = None
        if rank == 0:
            print(f"{chunks=} {test_chunks=}")
            print(f"{test_chunks.sum()=}", comm=comm, rank=0)
            # Calculate the difference between chunks and split sizes
            diff = np.sign(chunks - test_chunks)
            # The last rank shifts in reverse
            diff[-1] *= -1

            print(f"{diff=}")
            # Calculate new pivot
            new_pivots = np.sort(pivots + diff)

        pivots = Bcast(new_pivots, comm=comm, root=0)

        print(f"{pivots=}", comm=comm, rank=0)
