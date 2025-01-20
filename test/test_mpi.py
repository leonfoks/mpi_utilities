def test_bcast(world):
    ###
    mpiu.print("Broadcasting", world=world, rank=0)
    ###
    str = "broadcasted string" if world.rank == 0 else None
    str = mpiu.Bcast(str, world=world, root=0)
    assert str == "broadcasted string", ValueError("bcast str")
    world.barrier()

    i = 4 if world.rank == 0 else None
    j = mpiu.Bcast(i, world=world, root=0)
    assert isinstance(j, np.int64) and j == 4, TypeError("j must be np.int64")
    world.barrier()

    i = 5.0 if world.rank == 0 else None
    j = mpiu.Bcast(i, world=world, root=0)
    assert isinstance(j, np.float64) and j == 5.0, TypeError("j must be np.float64")
    world.barrier()

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = t(4) if world.rank == 0 else None
        j = mpiu.Bcast(i, world=world, root=0)
        assert isinstance(j, t) and j == 4, TypeError(f"j must be {t}")
        world.barrier()

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = np.arange(10, dtype=t) if world.rank == 0 else None
        j = mpiu.Bcast(i, world=world, root=0)
        assert np.all(j == np.arange(10, dtype=t)) and j.size == 10, ValueError(f"Bcast 1d {t}")
        world.barrier()

    i = np.zeros((2, 2), dtype=np.float64) if world.rank == 0 else None
    j = mpiu.Bcast(i, world=world, root=0)
    assert np.all(j == np.zeros((2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), ValueError("j")
    world.barrier()

    i = np.zeros((2, 2, 2), dtype=np.float64) if world.rank == 0 else None
    j = mpiu.Bcast(i, world=world, root=0)
    assert np.all(j == np.zeros((2, 2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), ValueError("j")
    world.barrier()

def test_send_recv(world):
    ###
    mpiu.print("Sending and receiving", world=world, rank=0)
    ###
    head_rank = world.rank == 0

    x = None; y = None
    if head_rank:
        x = 'Sent string'
        for i in range(1, world.size):
            mpiu.Send(x, dest=i, world=world)
    else:
        y = mpiu.Recv(source=0, world=world)
        assert y == 'Sent string', ValueError("send/recv string issue")
    world.barrier()


    i = None; j = None
    if head_rank:
        i = 4
        for k in range(1, world.size):
            mpiu.Send(i, dest=k, world=world)
    else:
        j = mpiu.Recv(source=0, world=world)
        assert isinstance(j, np.int64) and j == 4, TypeError("j must be np.int64")
    world.barrier()

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = None; j = None
        if head_rank:
            i = t(4)
            for k in range(1, world.size):
                mpiu.Send(i, dest=k, world=world)
        else:
            j = mpiu.Recv(source=0, world=world)
            assert isinstance(j, t) and j == 4, TypeError(f"send/recv scalar j must be {t}")
        world.barrier()

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = None; j = None
        if head_rank:
            i = np.arange(10, dtype=t)
            for k in range(1, world.size):
                mpiu.Send(i, dest=k, world=world)
        else:
            j = mpiu.Recv(source=0, world=world)
            assert np.all(j == np.arange(10, dtype=t)) and j.size == 10, TypeError(f"send/recv 1d j must be {t}")
        world.barrier()

    i = None; j = None
    if head_rank:
        i = np.zeros((2, 2), dtype=np.float64)
        for k in range(1, world.size):
            mpiu.Send(i, dest=k, world=world)
    else:
        j = mpiu.Recv(source=0, world=world)
        assert np.all(j == np.zeros((2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), TypeError(f"send/recv 2d")
    world.barrier()

    i = None; j = None
    if head_rank:
        i = np.zeros((2, 2, 2), dtype=np.float64)
        for k in range(1, world.size):
            mpiu.Send(i, dest=k, world=world)
    else:
        j = mpiu.Recv(source=0, world=world)
        assert np.all(j == np.zeros((2, 2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), TypeError(f"send/recv 3d")
    world.barrier()

def test_isend_irecv(world):
    ###
    mpiu.print("Isending and Ireceiving", world=world, rank=0)
    ###
    head_rank = world.rank == 0

    i = None; j = None
    if head_rank:
        i = 'Sent string'
        for k in range(1, world.size):
            mpiu.Isend(i, dest=k, world=world)
    else:
        j = mpiu.Irecv(source=0, world=world)
        assert j == 'Sent string', ValueError("isend/irecv string issue")
    world.barrier()

    i = None; j = None
    if head_rank:
        i = 4
        for k in range(1, world.size):
            mpiu.Isend(i, dest=k, world=world)
    else:
        j = mpiu.Irecv(source=0, world=world)
        assert isinstance(j, np.int64) and j == 4, TypeError("j must be np.int64")
    world.barrier()

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = None; j = None
        if head_rank:
            i = t(4)
            for k in range(1, world.size):
                mpiu.Isend(i, dest=k, world=world)
        else:
            j = mpiu.Irecv(source=0, world=world)
            assert isinstance(j, t) and j == 4, TypeError(f"isend/irecv scalar j must be {t}")
        world.barrier()

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = None; j = None
        if head_rank:
            i = np.arange(10, dtype=t)
            for k in range(1, world.size):
                mpiu.Isend(i, dest=k, world=world)
        else:
            j = mpiu.Irecv(source=0, world=world)
            assert np.all(j == np.arange(10, dtype=t)) and j.size == 10, TypeError(f"isend/irecv 1d j must be {t}")
        world.barrier()


    i = None; j = None
    if head_rank:
        i = np.zeros((2, 2), dtype=np.float64)
        for k in range(1, world.size):
            mpiu.Isend(i, dest=k, world=world)
    else:
        j = mpiu.Irecv(source=0, world=world)
        assert np.all(j == np.zeros((2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), TypeError(f"isend/irecv 2d j must be {t}")
    world.barrier()

    i = None; j = None
    if head_rank:
        i = np.zeros((2, 2, 2), dtype=np.float64)
        for k in range(1, world.size):
            mpiu.Isend(i, dest=k, world=world)
    else:
        j = mpiu.Irecv(source=0, world=world)
        assert np.all(j == np.zeros((2, 2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), TypeError(f"isend/irecv 3d j must be {t}")
    world.barrier()

def test_scatter(world):
    ###
    mpiu.print("Scatter", world=world, rank=0)
    ###

    n_test = np.int32(4 * world.size)
    starts, chunks = mpiu.load_balance(n_test, world.size)
    i0 = starts[world.rank]; i1 = i0 + chunks[world.rank]

    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        test = np.arange(n_test, dtype=t)[i0:i1]; i = None
        if world.rank == 0:
            N = np.int32(4 * world.size)
            i = np.arange(N, dtype=t)

        j = mpiu.Scatter(i, world)
        assert isinstance(j[0], t) and np.all(j == test) and j.size == chunks[world.rank], ValueError(f"Scatterv {t}")

def test_gather(world):
    ###
    mpiu.print("Gather", world=world, rank=0)
    ###

    # Gather scalars into 1d array
    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = t(world.rank)
        j = mpiu.Gather(i, world, root=0)
        if world.rank == 0:
            assert np.all(j == np.arange(world.size)) and isinstance(j[0], t) , ValueError(f"Gather scalars {t}")

    # Gather arrays into 1
    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = np.full(2, fill_value=world.rank, dtype=t)
        j = mpiu.Gather(i, world, root=0)
        if world.rank == 0:
            assert np.all(j == np.repeat(np.arange(world.size), 2)) and isinstance(j[0], t) , ValueError(f"Gatherv scalars {t}")

def test_allgather(world):
    ###
    mpiu.print("Allgather", world=world, rank=0)
    ###

    # Gather scalars into 1d array
    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = t(world.rank)
        j = mpiu.Allgather(i, world)
        assert np.all(j == np.arange(world.size)) and isinstance(j[0], t) , ValueError(f"Gather scalars {t}")

    # Gather arrays into 1
    for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        i = np.full(2, fill_value=world.rank, dtype=t)
        j = mpiu.Allgather(i, world)
        assert np.all(j == np.repeat(np.arange(world.size), 2)) and isinstance(j[0], t) , ValueError(f"Gatherv scalars {t}")

def test_scatterv(world):
    ###
    mpiu.print("Scatterv", world=world, rank=0)
    ###

    n_test = np.int32(4 * world.size + (0.5 * world.size))
    starts, chunks = mpiu.load_balance(n_test, world.size)
    i0 = starts[world.rank]; i1 = i0 + chunks[world.rank]

    for t in [np.int32, np.int64, np.float32, np.float64,np.complex128]:
        test = np.arange(n_test, dtype=t)[i0:i1]; i = None
        if world.rank == 0:
            N = np.int32(4 * world.size + (0.5 * world.size))
            i = np.arange(N, dtype=t)

        j = mpiu.Scatterv(i, world)
        assert isinstance(j[0], t) and np.all(j == test) and j.size == chunks[world.rank], ValueError(f"Scatterv {t}")

def test_gatherv(world):
    ###
    mpiu.print("Gatherv", world=world, rank=0)
    ###

    n_test = np.int32(4 * world.size + (0.5 * world.size))
    starts, chunks = mpiu.load_balance(n_test, world.size)
    i0 = starts[world.rank]; i1 = i0 + chunks[world.rank]
    test = np.arange(n_test)

    # Gather scalars into 1d array
    for t in [np.int32, np.int64, np.float32, np.float64,np.complex128]:
        i = t(world.rank)
        j = mpiu.Gatherv(i, world, root=0)
        if world.rank == 0:
            assert np.all(j == np.arange(world.size)) and isinstance(j[0], t) , ValueError(f"Gatherv scalars {t}")

    # Gather arrays into 1
    for t in [np.int32, np.int64, np.float32, np.float64,np.complex128]:
        i = np.arange(i0, i1, dtype=t)
        j = mpiu.Gatherv(i, world, root=0)
        if world.rank == 0:
            assert np.all(j == test) and isinstance(j[0], t) , ValueError(f"Gatherv 1d array {t}")

def test_allgatherv(world):
    ###
    mpiu.print("Allgatherv", world=world, rank=0)
    ###

    n_test = np.int32(4 * world.size + (0.5 * world.size))
    starts, chunks = mpiu.load_balance(n_test, world.size)
    i0 = starts[world.rank]; i1 = i0 + chunks[world.rank]
    test = np.arange(n_test)

    # Gather scalars into 1d array
    for t in [np.int32, np.int64, np.float32, np.float64,np.complex128]:
        i = t(world.rank)
        j = mpiu.Allgatherv(i, world)
        assert np.all(j == np.arange(world.size)) and isinstance(j[0], t) , ValueError(f"Gatherv scalars {t}")

    # Gather arrays into 1
    for t in [np.int32, np.int64, np.float32, np.float64,np.complex128]:
        i = np.arange(i0, i1, dtype=t)
        j = mpiu.Allgatherv(i, world)
        assert np.all(j == test) and isinstance(j[0], t) , ValueError(f"Gatherv 1d array {t}")

def test_reduce(world):
    ###
    mpiu.print("Reducing", world=world, rank=0)
    ###
    head_rank = world.rank == 0

    i = None; j = None
    i = 4
    j = mpiu.Reduce(i, 'sum', world)

    if head_rank:
        assert j == 4*world.size, TypeError("j must be np.int64")
    world.barrier()

    for t in [np.float32, np.float64, np.complex128]:
        i = None; j = None
        i = t(4)
        j = mpiu.Reduce(i, 'sum', world)
        print(j.dtype)
        print(isinstance(j.dtype, t))
        if head_rank:
            assert isinstance(j, t) and j == 4*world.size, TypeError(f"isend/irecv scalar j must be {t}")
        world.barrier()

    # for t in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
    #     i = None; j = None
    #     if head_rank:
    #         i = np.arange(10, dtype=t)
    #         for k in range(1, world.size):
    #             mpiu.Isend(i, dest=k, world=world)
    #     else:
    #         j = mpiu.Irecv(source=0, world=world)
    #         assert np.all(j == np.arange(10, dtype=t)) and j.size == 10, TypeError(f"isend/irecv 1d j must be {t}")
    #     world.barrier()


    # i = None; j = None
    # if head_rank:
    #     i = np.zeros((2, 2), dtype=np.float64)
    #     for k in range(1, world.size):
    #         mpiu.Isend(i, dest=k, world=world)
    # else:
    #     j = mpiu.Irecv(source=0, world=world)
    #     assert np.all(j == np.zeros((2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), TypeError(f"isend/irecv 2d j must be {t}")
    # world.barrier()

    # i = None; j = None
    # if head_rank:
    #     i = np.zeros((2, 2, 2), dtype=np.float64)
    #     for k in range(1, world.size):
    #         mpiu.Isend(i, dest=k, world=world)
    # else:
    #     j = mpiu.Irecv(source=0, world=world)
    #     assert np.all(j == np.zeros((2, 2, 2), dtype=np.float64)) and np.all(np.asarray(np.shape(j)) == 2), TypeError(f"isend/irecv 3d j must be {t}")
    # world.barrier()


if __name__ == "__main__":

    import numpy as np
    from mpi4py import MPI
    import mpi4py_utilities as mpiu

    ##############################################################################
    # Init
    ##############################################################################
    world = MPI.COMM_WORLD

    mpiu.hello_world(world)
    world.barrier()

    test_bcast(world)
    test_send_recv(world)
    test_isend_irecv(world)
    test_scatter(world)
    test_gather(world)
    test_allgather(world)
    test_scatterv(world)
    test_gatherv(world)
    test_allgatherv(world)
    test_reduce(world)