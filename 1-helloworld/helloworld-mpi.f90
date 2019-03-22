program helloworld
    implicit none
    include 'mpif.h'

    INTEGER :: rank
    INTEGER :: size
    INTEGER :: ierror
    INTEGER :: tag
    INTEGER :: status(MPI_STATUS_SIZE)

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

    print *, 'Hello world! from process ', rank

    call MPI_FINALIZE(ierror)
    
end program helloworld