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
        subroutine FillMatrix(matrixA, matrixB, size)
            use precision
            REAL(long), INTENT(OUT) :: matrixA(:)
            REAL(long), INTENT(OUT) :: matrixB(:)
            INTEGER, INTENT(IN) :: size
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
            use precision
            REAL(long), INTENT(IN) :: MatrixA(:)
            REAL(long), INTENT(IN) :: MatrixB(:)
            REAL(long), INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: size
        end subroutine Multiply

        subroutine AddMatrix(matrix, size, result)
            use precision
            REAL(long), INTENT(IN) :: matrix(:)
            INTEGER, INTENT(IN) :: size
            REAL(long), INTENT(OUT) :: result
        end subroutine AddMatrix
    END INTERFACE

    INTEGER :: size
    REAL(long), POINTER, DIMENSION(:) :: matrixA
    REAL(long), POINTER, DIMENSION(:) :: matrixB
    REAL(long), POINTER, DIMENSION(:) :: matrixC
    REAL(long) :: result

    size = 5

    allocate(matrixA(size * size))
    allocate(matrixB(size * size))
    allocate(matrixC(size * size))

    call FillMatrix(matrixA,matrixB, size)
    call Multiply(MatrixA, MatrixB, MatrixC, size)
    call AddMatrix(MatrixC, size, result)

    write(*, *) 'result calc:', size * size * size * a * a
    write(*, *) 'result:', result

    deallocate(matrixA)
    deallocate(matrixB)
    deallocate(matrixC)

end program matmult

subroutine FillMatrix(matrixA, matrixB, size)
    use precision
    use parameters
    IMPLICIT NONE
    REAL(long), INTENT(OUT) :: matrixA(:)
    REAL(long), INTENT(OUT) :: matrixB(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i
    INTEGER :: j

    do i = 1, size, 1
        do j = 1, size, 1
            matrixA((i - 1) * size + j)  = a
            matrixB((i - 1) * size + j)  = a
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: MatrixA(:)
    REAL(long), INTENT(IN) :: MatrixB(:)
    REAL(long), INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i
    INTEGER :: j
    INTEGER :: k
    REAL(long) :: result

    do i = 1, size, 1
        do j = 1, size, 1
            result = 0.0
            do k = 1, size, 1
                result = result + matrixA((i - 1) * size + k) * matrixB((k - 1) * size + j)
            end do
            matrixC((i - 1) * size + j) = result
        end do
    end do

end subroutine Multiply

subroutine AddMatrix(matrix, size, result)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: matrix(:)
    INTEGER, INTENT(IN) :: size
    REAL(long), INTENT(OUT) :: result
    INTEGER :: i
    INTEGER :: j

    result = 0.0_long

    do i = 1, size, 1
        do j = 1, size, 1
            result = result + matrix((i - 1) * size + j)
        end do
    end do

end subroutine AddMatrix