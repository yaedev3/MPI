! File: matmult.f90
!
! Purpose: 
! 
! Input:
! 
! Output:
! 
! Compile: gfortran -o matmult-f90.o matmult.f90
! 
! Run: ./matmult-f90.o
! 
! Algorithm:
! 
! Note:

program matmult
    implicit none
    
    INTERFACE 
        subroutine PrintMatrix(array, size)
            REAL, INTENT(IN) :: array(:)
            INTEGER, INTENT(IN) :: size
        end subroutine PrintMatrix

        subroutine FillMatrix(array, size)
            REAL, INTENT(OUT) :: array(:)
            INTEGER, INTENT(IN) :: size
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
            REAL, INTENT(IN) :: MatrixA(:), MatrixB(:)
            REAL, INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: size
        end subroutine Multiply

    END INTERFACE

    INTEGER :: size = 2
    REAL, DIMENSION(4) :: matrixA, matrixB, matrixC;

    call FillMatrix(matrixA, size)
    call FillMatrix(matrixB, size)

    call PrintMatrix(matrixA, size)
    call PrintMatrix(matrixB, size)

    call Multiply(MatrixA, MatrixB, MatrixC, size)
    call PrintMatrix(matrixC, size)

end program matmult

subroutine PrintMatrix(array, size)
    IMPLICIT NONE
    REAL, INTENT(IN) :: array(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i, j

    do i = 0, size - 1, 1
        do j = 0, size - 1, 1
            write (*,*) i, j, array(i * size + j) 
        end do
    end do

end subroutine PrintMatrix

subroutine FillMatrix(array, size)
    IMPLICIT NONE
    REAL, INTENT(OUT) :: array(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i, j

    do i = 0, size - 1, 1
        do j = 0, size, 1
            array(i * size + j)  = rand(i)
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
    IMPLICIT NONE
    REAL, INTENT(IN) :: MatrixA(:), MatrixB(:)
    REAL, INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i, j, k
    REAL :: result

    do i = 0, size - 1, 1
        do j = 0, size - 1, 1
            result = 0
            do k = 0, size - 1, 1
                result = result + matrixA(i * size + k) * matrixB(k * size + j)
            end do
            matrixC(i * size + j) = result
        end do
    end do

end subroutine Multiply