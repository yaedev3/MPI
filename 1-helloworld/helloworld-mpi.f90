! File: helloworld-mpi.f90
!
! Purpose: 
! 
! Input:
! 
! Output:
! 
! Compile: mpif90 -o helloworld-mpi-f90.o helloworld-mpi.f90
! 
! Run: mpiexec -np 10 ./helloworld-mpi-f90.o
! 
! Algorithm:
! 
! Note:

program helloworld
implicit none
include 'mpif.h'

CHARACTER *20 msg
INTEGER rank, size, ierror, tag, status(MPI_STATUS_SIZE)

call MPI_INIT(ierror)
call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)
print*, 'Hello world! from process ', rank
call MPI_FINALIZE(ierror)
    
end program helloworld