! File: matmult-mpi.f90
!
! Purpose: 
! 
! Input:
! 
! Output:
! 
! Compile: mpif90_intel -o matmult-mpi-f90.o matmult-mpi.f90
! 
! Run: mpiexec_intel -np 10 ./matmult-mpi-f90.o
! 
! Algorithm:
! 
! Note:

program matmult
    implicit none
    include 'mpif.h'

    INTERFACE 
    subroutine PrintMatrix(matrix, N, name)
        REAL, INTENT(IN) :: matrix(:)
        INTEGER, INTENT(IN) :: N
        CHARACTER(len=*), INTENT(IN) :: name
    end subroutine PrintMatrix

        subroutine FillMatrix(array, N)
            REAL, INTENT(OUT) :: array(:)
            INTEGER, INTENT(IN) :: N
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, NX, NY)
            REAL, INTENT(IN) :: MatrixA(:)
            REAL, INTENT(IN) :: MatrixB(:)
            REAL, INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: NX
            INTEGER, INTENT(IN) :: NY
        end subroutine Multiply

    END INTERFACE

    INTEGER :: N
    REAL, POINTER, DIMENSION(:) :: matrixA
    REAL, POINTER, DIMENSION(:) :: matrixB
    REAL, POINTER, DIMENSION(:) :: matrixC
    INTEGER :: rank 
    INTEGER :: process
    INTEGER :: processN
    INTEGER :: ierror
    INTEGER :: status(MPI_STATUS_SIZE)
    INTEGER :: waste
    INTEGER :: n_local
    INTEGER :: i

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, process, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

    N = 8
    waste = mod(N, process - 1)
    n_local = N / (process - 1)
    allocate(matrixB(N * N))

    if (rank == 0) then

        allocate(matrixA(N * N))
        allocate(matrixC(N * N))
        call FillMatrix(matrixA, N)
        call FillMatrix(matrixB, N)

    end if

    call MPI_BCAST(matrixB, N * N, MPI_REAL, 0, MPI_COMM_WORLD, ierror)

    if (rank == 0) then

        processN = n_local
        do i = 1, process - 1, 1

            if (i == process - 1) then
                processN = waste + n_local
            end if

            call MPI_SEND(matrixA((i - 1) * N * n_local + 1), processN * N, MPI_REAL, i, 0, MPI_COMM_WORLD, ierror)

        end do

        processN = n_local
        do i = 1, process - 1, 1

            if (i == process - 1) then
                processN = waste + n_local
            end if

            call MPI_RECV(matrixC((i - 1) * N * n_local + 1), processN * N, MPI_REAL, i, 1, MPI_COMM_WORLD, status, ierror)

        end do

        call PrintMatrix(matrixA, N, 'Matrix A')
        call PrintMatrix(matrixB, N, 'Matrix B')
        call PrintMatrix(matrixC, N, 'Matrix C (result)')

    else 

        if (rank == process - 1) then
            processN = waste + n_local
        else
            processN = n_local
        end if

        allocate(matrixA(processN * N))
        allocate(matrixC(processN * N))

        call MPI_RECV(matrixA, processN * N, MPI_REAL, 0, 0, MPI_COMM_WORLD, status, ierror)

        call Multiply(matrixA, matrixB, matrixC, processN, N)

        call MPI_SEND(matrixC, processN * N, MPI_REAL, 0, 1, MPI_COMM_WORLD, ierror)

    end if

    deallocate(matrixA);
    deallocate(matrixB);
    deallocate(matrixC);

    call MPI_FINALIZE(ierror)

end program matmult

subroutine PrintMatrix(matrix, N, name)
    IMPLICIT NONE
    REAL, INTENT(IN) :: matrix(:)
    INTEGER, INTENT(IN) :: N
    CHARACTER(len=*), INTENT(IN) :: name
    INTEGER :: i
    INTEGER :: j

    write(*, *) name
    do i = 1, N, 1
        do j = 1, N, 1
            write (*,*) i, j, matrix((i - 1) * N + j) 
        end do
    end do

end subroutine PrintMatrix

subroutine FillMatrix(matrix, N)
    IMPLICIT NONE
    REAL, INTENT(OUT) :: matrix(:)
    INTEGER, INTENT(IN) :: N
    INTEGER :: i
    INTEGER :: j

    do i = 1, N, 1
        do j = 1, N, 1
            matrix((i - 1) * N + j)  = 1.0
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, NX, NY)
    IMPLICIT NONE
    REAL, INTENT(IN) :: MatrixA(:)
    REAL, INTENT(IN) :: MatrixB(:)
    REAL, INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: NX
    INTEGER, INTENT(IN) :: NY
    INTEGER :: i
    INTEGER :: j
    INTEGER :: k
    REAL :: result

    do i = 1, NX, 1
        do j = 1, NY, 1

            result = 0.0
            do k = 1, NY, 1
                result = result + matrixA((i - 1) * NY + k) * matrixB((k - 1) * NY + j)
            end do
            matrixC((i - 1) * NY + j) = result
            
        end do
    end do

end subroutine Multiply
