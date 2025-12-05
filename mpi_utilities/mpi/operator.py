def operator(this):
    """Get the corresponding MPI operator

    """
    from mpi4py import MPI

    match this.lower():
        case 'max':
            return MPI.MAX
        case 'min':
            return MPI.MIN
        case 'sum':
            return MPI.SUM
        case 'prod':
            return MPI.PROD
        case 'land':
            return MPI.LAND
        case 'lor':
            return MPI.LOR
        case 'band':
            return MPI.BAND
        case 'bor':
            return MPI.BOR
        case 'maxloc':
            return MPI.MAXLOC
        case 'minloc':
            return MPI.MINLOC
        case _:
            assert False, ValueError(f"this {this} must be from (max, min, sum, prod, land, lor, band, bor, maxloc, minloc)")