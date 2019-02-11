MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

program matmult
    use precision
    implicit none
    
    INTERFACE 
        subroutine PrintMatrix(matrix, size, name)
            REAL, INTENT(IN) :: matrix(:)
            INTEGER, INTENT(IN) :: size
            CHARACTER(len=*), INTENT(IN) :: name
        end subroutine PrintMatrix

        subroutine FillMatrix(matrixA, matrixB, size)
            REAL, INTENT(OUT) :: matrixA(:)
            REAL, INTENT(OUT) :: matrixB(:)
            INTEGER, INTENT(IN) :: size
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
            REAL, INTENT(IN) :: MatrixA(:)
            REAL, INTENT(IN) :: MatrixB(:)
            REAL, INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: size
        end subroutine Multiply

    END INTERFACE

    INTEGER, PARAMETER :: size = 5
    REAL, DIMENSION(size * size) :: matrixA
    REAL, DIMENSION(size * size) :: matrixB
    REAL, DIMENSION(size * size) :: matrixC

    call FillMatrix(matrixA,matrixB, size)

    call Multiply(MatrixA, MatrixB, MatrixC, size)

    call PrintMatrix(matrixA, size, 'Matrix A')
    call PrintMatrix(matrixB, size, 'Matrix B')
    call PrintMatrix(matrixC, size, 'Matrix C (result)')

end program matmult

subroutine PrintMatrix(matrix, size, name)
    IMPLICIT NONE
    REAL, INTENT(IN) :: matrix(:)
    INTEGER, INTENT(IN) :: size
    CHARACTER(len=*), INTENT(IN) :: name
    INTEGER :: i
    INTEGER :: j

    write(*, *) name
    do i = 1, size, 1
        do j = 1, size, 1
            write (*,*) i, j, matrix((i - 1) * size + j) 
        end do
    end do

end subroutine PrintMatrix

subroutine FillMatrix(matrixA, matrixB, size)
    use precision
    IMPLICIT NONE
    REAL, INTENT(OUT) :: matrixA(:)
    REAL, INTENT(OUT) :: matrixB(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i
    INTEGER :: j

    do i = 1, size, 1
        do j = 1, size, 1
            matrixA((i - 1) * size + j)  = 1.0E-10_long
            matrixB((i - 1) * size + j)  = 1.0E-10_long
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
    IMPLICIT NONE
    REAL, INTENT(IN) :: MatrixA(:)
    REAL, INTENT(IN) :: MatrixB(:)
    REAL, INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i
    INTEGER :: j
    INTEGER :: k
    REAL :: result

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