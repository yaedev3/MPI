MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

MODULE parameters
  USE precision
  IMPLICIT NONE
  REAL(long), PARAMETER :: a = 1.0E-10_long         ! Valor constante para llenar la matriz
END MODULE parameters

module time_ftcs
    contains
      function timestamp() result(str)
        implicit none
        character(len=20)                     :: str
        integer                               :: values(8)
        character(len=4)                      :: year
        character(len=2)                      :: month
        character(len=2)                      :: day, hour, minute, second
        character(len=5)                      :: zone
    
        ! Get current time
        call date_and_time(VALUES=values, ZONE=zone)  
        write(year,'(i4.4)')    values(1)
        write(month,'(i2.2)')   values(2)
        write(day,'(i2.2)')     values(3)
        write(hour,'(i2.2)')    values(5)
        write(minute,'(i2.2)')  values(6)
        write(second,'(i2.2)')  values(7)
    
        str = day//'-'//hour//'-'//minute//'-'//second
      end function timestamp
end module

program matmult
    use precision
    use parameters
    use time_ftcs
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

    INTEGER :: N                                        ! Dimension de la matriz
    REAL(long), POINTER, DIMENSION(:) :: matrixA        ! Primera matriz
    REAL(long), POINTER, DIMENSION(:) :: matrixB        ! Segunda matriz
    REAL(long), POINTER, DIMENSION(:) :: matrixC        ! Matriz resultado
    REAL(long) :: result                                ! Resultado de la suma de la matriz resultado
    REAL(long) :: estimation                            ! Estimacion del calculo
    REAL(long) :: error                                 ! Error encontrado
    REAL(long) :: elapsed                               ! Tiempo que tomo ejecutarse el programa
    REAL(long) :: start_clock                           ! Tiempo de inicio
    REAL(long) :: stop_clock                            ! Tiempo de termino
    CHARACTER(len=38) :: start                          ! Hora de inicio
    CHARACTER(len=38) :: end                            ! Hora de termino

    ! Establece la hora de inicio
    start = timestamp()
    call CPU_TIME(start_clock)

    ! Asigna la dimension de la matriz
    call OpenFile(N)

    ! Reserva la memoria para las tres matrices
    allocate(matrixA(N * N))
    allocate(matrixB(N * N))
    allocate(matrixC(N * N))

    ! Llena la matriz A y B con el valor constante
    call FillMatrix(matrixA,matrixB, N)
    
    ! Multiplica las matrices A y B guardando el valor en la matriz C
    call Multiply(MatrixA, MatrixB, MatrixC, N)
    
    ! Calcula la suma de los valores de la matriz C
    call AddMatrix(MatrixC, N, result)

    ! Calculo estimado con la formula a^2*N^3.
    estimation = N ** 3_long * a ** 2_long

    ! Calcula el % de error
    error = DABS(result - estimation) / estimation * 100.0_long

    ! Libera la memoria de las tres matrices
    deallocate(matrixA)
    deallocate(matrixB)
    deallocate(matrixC)

    ! Establece la hora en que termino de calcular
    end = timestamp()
    call CPU_TIME(stop_clock)

    ! Calcula el tiempo que tomo ejecutar el programa
    elapsed =  stop_clock - start_clock

    ! Guarda el resultado en un archivo
    call SaveFile(start, end, error, elapsed, N)

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

! Abre un archivo con la dimension de la matriz
subroutine OpenFile(N)
    implicit none
    INTEGER, INTENT(OUT) :: N               ! Dimension de la matriz

    OPEN(UNIT=1,file='parameters.dat',ACTION="READ",STATUS='OLD')
    READ(1,*) N
    CLOSE(1)

end subroutine OpenFile

! Crea un archivo de salida con la hora de inicio, de termino y el tiempo que tomo correr el programa
! Asi como el porcentaje de error
subroutine SaveFile(start, end, error, elapsed, N)
    use precision
    CHARACTER(len=38), INTENT(IN) :: start
    CHARACTER(len=38), INTENT(IN) :: end
    REAL(long), INTENT(IN) :: error
    REAL(long), INTENT(IN) :: elapsed
    INTEGER, INTENT(IN) :: N
    CHARACTER(len = 54) :: file_name
    CHARACTER(len = 4) :: Nstring

    write(Nstring, "(I4)") N 

    file_name = 'serial-f90-'//Nstring//'-'//trim(end)//'.txt'
    
    open (unit=10,file=file_name)
    write(10,*) 'Hora de inicio'
    write(10,*) start
    write(10,*) 'Hora de termino'
    write(10,*) end
    write(10,*) 'Tiempo de ejecucion'
    write(10,*) elapsed
    write(10,*) 'Error'
    write(10,*) error
    close(10)

end subroutine SaveFile