import numpy as np
from mpi4py import MPI
from ..mpi.Allgather import Allgather
from ..chunking import starts_from_chunks

def rechunk(values, chunks, *, comm, **kwargs):

    rank = comm.rank
    size = comm.size
    dtype = values.dtype
    nv = values.size
    ranks = np.arange(size)

    current_chunks = Allgather(nv, comm=comm)

    if np.all(current_chunks == chunks):
        return values

    # Get the starting indices for the current and new chunks
    current_starts = starts_from_chunks(current_chunks)
    new_starts = starts_from_chunks(chunks)
    new_chunk = chunks[rank]

    # We will send and recv possibly multiple boundaries from each rank to other ranks
    # Create lists of Isend and Irecv requests
    sends = []; recvs = []

    # Create output array for rechunked data
    out = np.empty(new_chunk, dtype=dtype)

    my_start = current_starts[rank]

    # Get boundaries to send to other ranks
    boundary_starts = np.maximum(new_starts, my_start)
    boundary_ends = np.minimum(new_starts + chunks, my_start + nv)
    condition = boundary_starts < boundary_ends

    # Trim array to only those that need sending
    dests = ranks[condition]
    new_starts = new_starts[condition]
    boundary_starts = boundary_starts[condition]
    boundary_ends = boundary_ends[condition]

    # Send out boundary information
    for i in range(dests.size):
        new_start = new_starts[i]
        # boundary between myself and the other rank
        boundary_start = boundary_starts[i]
        dest = dests[i]

        # This is either the boundary to send or my ranks maintained region
        send_this = values[boundary_start - my_start:boundary_ends[i] - my_start]

        # I am myself, just assign the core region that doesnt need sending
        if dest == rank:
            recv_start = boundary_start - new_start
            out[recv_start:recv_start + send_this.size] = send_this
        else:
            # ISend boundary to the other rank.
            # This fires off the sends without worrying about receiving
            sends.append(comm.Isend(send_this, dest=dest, tag=0))

    # Recieve boundary information for my rank
    new_start = starts_from_chunks(chunks)[rank]

    boundary_starts = np.maximum(current_starts, new_start)
    boundary_ends = np.minimum(current_starts + current_chunks, new_start + chunks[rank])
    condition = (boundary_starts < boundary_ends) & (ranks != rank)

    boundary_starts = boundary_starts[condition]
    boundary_ends = boundary_ends[condition]
    sources = ranks[condition]

    for i in range(sources.size):
        boundary_start = boundary_starts[i]
        # Create the boundary buffer
        recv_this = np.empty(boundary_ends[i] - boundary_start, dtype=dtype)
        # Irecv boundary from the other rank.
        # This fires off the recvs without worrying about sending
        # We need to keep the request, the start index of this boundary, and the memory reference to the boundary itself
        recvs.append((comm.Irecv(recv_this, source=sources[i], tag=0), boundary_start - new_start, recv_this))

    # Wait for the Irecvs and combine boundaries into the rechunked array
    for req, start, vals in recvs:
        req.Wait()
        out[start:start + vals.size] = vals

    # Wait for the Isends
    MPI.Request.Waitall(sends)

    # Return the rechunked array
    return out