MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

MODULE parameters
  USE precision
  IMPLICIT NONE
  REAL(long), PARAMETER :: a = 1.0E-10_long         ! Valor constante para llenar la matriz
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

    INTEGER, PARAMETER :: N  = 5                        ! Dimension de la matriz
    REAL(long), DIMENSION(N * N) :: matrixA             ! Primera matriz
    REAL(long), DIMENSION(N * N) :: matrixB             ! Segunda matriz
    REAL(long), DIMENSION(N * N) :: matrixC             ! Matriz resultado
    REAL(long) :: result                                ! Resultado de la suma de la matriz resultado
    REAL(long) :: estimation                            ! Estimacion del calculo
    REAL(long) :: error                                 ! Error encontrado


    ! Llena la matriz A y B con el valor constante
    call FillMatrix(matrixA,matrixB, N)
    
    ! Multiplica las matrices A y B guardando el valor en la matriz C
    call Multiply(MatrixA, MatrixB, MatrixC, N)
    
    ! Calcula la suma de los valores de la matriz C
    call AddMatrix(MatrixC, N, result)

    ! Calculo estimado con la formula a^2*N^3.
    estimation = N ** 3_long * a ** 2_long

    ! Calcula el % de error
    error = DABS(result - estimation) / estimation * 100.0_long;

    ! Imprime el % de error
    write(*, *) 'result ', error, "N = ", N

end program matmult

! Llena las dos matrices con el valor constante
subroutine FillMatrix(matrixA, matrixB, N)
    use precision
    use parameters
    IMPLICIT NONE
    REAL(long), INTENT(OUT) :: matrixA(:)   ! Primera matriz
    REAL(long), INTENT(OUT) :: matrixB(:)   ! Segunda matriz
    INTEGER, INTENT(IN) :: N                ! Dimension de la matriz
    INTEGER :: i                            ! Indice el renglon
    INTEGER :: j                            ! Indice de la columna

    do i = 1, N, 1
        do j = 1, N, 1
            matrixA((i - 1) * N + j)  = a
            matrixB((i - 1) * N + j)  = a
        end do
    end do

end subroutine FillMatrix

! Multiplica las dos matrices y almacena el resultado en la matriz de resultado
subroutine Multiply(MatrixA, MatrixB, MatrixC, N)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: MatrixA(:)    ! Primera matriz
    REAL(long), INTENT(IN) :: MatrixB(:)    ! Segunda matriz
    REAL(long), INTENT(OUT) :: MatrixC(:)   ! Matriz resultado
    INTEGER, INTENT(IN) :: N                ! Dimension de la matriz
    INTEGER :: i                            ! Indice del renglon
    INTEGER :: j                            ! Indice de la columna
    INTEGER :: k                            ! Indice de la multiplicacion
    REAL(long) :: result                    ! Resultado de la multiplicacion

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

! Suma todos los elementos de una matriz y regresa el resultado
subroutine AddMatrix(matrix, N, result)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: matrix(:)     ! Matriz resultado
    INTEGER, INTENT(IN) :: N                ! Dimension de la matriz
    REAL(long), INTENT(OUT) :: result       ! Resultado de la suma
    INTEGER :: i                            ! Indice del renglon
    INTEGER :: j                            ! Indice de la columna

    result = 0.0_long

    do i = 1, N, 1
        do j = 1, N, 1
            result = result + matrix((i - 1) * N + j)
        end do
    end do

end subroutine AddMatrix
