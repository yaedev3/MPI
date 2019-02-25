MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

MODULE parameters
  USE precision
  IMPLICIT NONE
  REAL(long), PARAMETER :: a = 1.0E-10_long
END MODULE parameters

program matmult
    use precision
    use parameters
    implicit none
    
    INTERFACE 
        subroutine FillMatrix(matrixA, matrixB, N)
            use precision
            REAL(long), INTENT(OUT) :: matrixA(:)
            REAL(long), INTENT(OUT) :: matrixB(:)
            INTEGER, INTENT(IN) :: N
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, N)
            use precision
            REAL(long), INTENT(IN) :: MatrixA(:)
            REAL(long), INTENT(IN) :: MatrixB(:)
            REAL(long), INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: N
        end subroutine Multiply

        subroutine AddMatrix(matrix, N, result)
            use precision
            REAL(long), INTENT(IN) :: matrix(:)
            INTEGER, INTENT(IN) :: N
            REAL(long), INTENT(OUT) :: result
        end subroutine AddMatrix
    END INTERFACE

    INTEGER :: N
    REAL(long), POINTER, DIMENSION(:) :: matrixA
    REAL(long), POINTER, DIMENSION(:) :: matrixB
    REAL(long), POINTER, DIMENSION(:) :: matrixC
    REAL(long) :: result
    REAL(long) :: estimation
    REAL(long) :: error

    N = 4096

    allocate(matrixA(N * N))
    allocate(matrixB(N * N))
    allocate(matrixC(N * N))

    call FillMatrix(matrixA,matrixB, N)
    call Multiply(MatrixA, MatrixB, MatrixC, N)
    call AddMatrix(MatrixC, N, result)

    estimation = N ** 3_long * a ** 2_long

    error = DABS(result - estimation) / estimation * 100.0_long;

    write(*, *) 'result ', error, "N = ", N

    deallocate(matrixA)
    deallocate(matrixB)
    deallocate(matrixC)

end program matmult

subroutine FillMatrix(matrixA, matrixB, N)
    use precision
    use parameters
    IMPLICIT NONE
    REAL(long), INTENT(OUT) :: matrixA(:)
    REAL(long), INTENT(OUT) :: matrixB(:)
    INTEGER, INTENT(IN) :: N
    INTEGER :: i
    INTEGER :: j

    do i = 1, N, 1
        do j = 1, N, 1
            matrixA((i - 1) * N + j)  = a
            matrixB((i - 1) * N + j)  = a
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, N)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: MatrixA(:)
    REAL(long), INTENT(IN) :: MatrixB(:)
    REAL(long), INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: N
    INTEGER :: i
    INTEGER :: j
    INTEGER :: k
    REAL(long) :: result

    do i = 1, N, 1
        do j = 1, N, 1
            result = 0.0
            do k = 1, N, 1
                result = result + matrixA((i - 1) * N + k) * matrixB((k - 1) * N + j)
            end do
            matrixC((i - 1) * N + j) = result
        end do
    end do

end subroutine Multiply

subroutine AddMatrix(matrix, N, result)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: matrix(:)
    INTEGER, INTENT(IN) :: N
    REAL(long), INTENT(OUT) :: result
    INTEGER :: i
    INTEGER :: j

    result = 0.0_long

    do i = 1, N, 1
        do j = 1, N, 1
            result = result + matrix((i - 1) * N + j)
        end do
    end do

end subroutine AddMatrix
