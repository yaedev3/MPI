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
    include 'mpif.h'
    
    INTERFACE 
        subroutine FillMatrix(matrixA, matrixB, N)
            use precision
            REAL(long), INTENT(OUT) :: matrixA(:)
            REAL(long), INTENT(OUT) :: matrixB(:)
            INTEGER, INTENT(IN) :: N
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, sizeX, sizeY)
            use precision
            REAL(long), INTENT(IN) :: MatrixA(:)
            REAL(long), INTENT(IN) :: MatrixB(:)
            REAL(long), INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: sizeX
            INTEGER, INTENT(IN) :: sizeY
        end subroutine Multiply

        subroutine AddMatrix(matrix, Nx, Ny, result)
            use precision
            REAL(long), INTENT(IN) :: matrix(:)
            INTEGER, INTENT(IN) :: Nx
            INTEGER, INTENT(IN) :: Ny
            REAL(long), INTENT(OUT) :: result
        end subroutine AddMatrix

        subroutine OpenFile(N)
            INTEGER, INTENT(OUT) :: N
        end subroutine OpenFile
    END INTERFACE

    INTEGER :: N                                    ! Dimension de la matriz
    REAL(long), POINTER, DIMENSION(:) :: matrixA    ! Primera matriz
    REAL(long), POINTER, DIMENSION(:) :: matrixB    ! Segunda matriz
    REAL(long), POINTER, DIMENSION(:) :: matrixC    ! Matriz resultado
    REAL(long) :: result                            ! Resultado de la suma de la matriz resultado
    REAL(long) :: result_local                      ! Resultado de la suma local de cada proceso
    REAL(long) :: estimation                        ! Estimacion del calculo
    REAL(long) :: error                             ! Error encontrado
    INTEGER :: rank                                 ! Indice de cada proceso
    INTEGER :: process                              ! Numero total de procesos
    INTEGER :: processSize                          ! Tamaño corregido
    INTEGER :: ierror                               ! Error con MPI
    INTEGER :: status(MPI_STATUS_SIZE)              ! Estado de MPI
    INTEGER :: waste                                ! Residuo de informacion
    INTEGER :: n_local                              ! Tamaño de informacion por proceso
    INTEGER :: i                                    ! Iterador de procesos
 
    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, process, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

    if (rank == 0) then

        ! Asigna la dimension de la matriz
        call OpenFile(N)
        
    end if

    ! Comparte la dimension de la matriz
    call MPI_BCAST(N, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, ierror)
    
    ! Calcula el sobrante de datos en caso de que los procesos no sean
    ! multiplos de la informacion
    waste = mod(N, process - 1)

    ! Calcula el numero de datos que va a tomar cada proceso
    n_local = N / (process - 1)

    ! Inicializa la matriz B para todos los procesos
    allocate(matrixB(N * N))

    if (rank == 0) then

        ! Inicializa matriz A con el tamaño total (N * N)
        allocate(matrixA(N * N))

        ! Inicializa el resultado en 0.0
        result = 0.0_long

        ! Llena la matriz A y B con el valor constante
        call FillMatrix(matrixA, matrixB, N)

    end if

    ! Comparte la informacion de la matriz B con los demas procesos
    call MPI_BCAST(matrixB, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD, ierror)

    if (rank == 0) then
        
        ! Envia partes de la matriz A a todos los demas procesos
        processSize = n_local
        do i = 1, process - 1, 1

            if (i == process - 1) then
                processSize = waste + n_local
            end if

            call MPI_SEND(matrixA((i - 1) * N * n_local + 1), processSize * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, ierror)

        end do

        ! Recibe el resultado de la suma de todos los elementos de la matriz C
        do i = 1, process - 1, 1

            call MPI_RECV(result_local, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, status, ierror)
            result = result + result_local

        end do

        ! Calculo estimado con la formula a^2*N^3
        estimation = N ** 3_long * a ** 2_long

        ! Calcula el % de error
        error = DABS(result - estimation) / estimation * 100.0_long;

        ! Imprime el % de error
        write(*, *) 'result ', error, "N = ", N

    else 
        ! Verifica si es el ultimo proceso para calcular el reciduo
        if (rank == process - 1) then
            processSize = waste + n_local;
        else
            processSize = n_local;
        end if

        ! Inicializa las matrices A y C con los tamaños correspondientes a cada proceso
        allocate(matrixA(N * processSize))
        allocate(matrixC(N * processSize))

        ! Recibe la matriz A del proceso 0
        call MPI_RECV(matrixA, processSize * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, status, ierror)

        ! Hace la multiplicacion de matrices.
        call Multiply(matrixA, matrixB, matrixC, processSize, N)

        ! Hace la suma de todos los elementos de la matriz C
        call AddMatrix(MatrixC, processSize, N, result_local)

        ! Envia el resultado de la multiplicacion al proceso 0
        call MPI_SEND(result_local, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, ierror)

        ! Libera la memoria de la matriz C
        deallocate(matrixC)

    end if

    ! Libera la memoria de las matrices A y B
    deallocate(matrixA)
    deallocate(matrixB)

    call MPI_FINALIZE(ierror)

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
subroutine Multiply(MatrixA, MatrixB, MatrixC, sizeX, sizeY)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: MatrixA(:)    ! Primera matriz
    REAL(long), INTENT(IN) :: MatrixB(:)    ! Segunda matriz
    REAL(long), INTENT(OUT) :: MatrixC(:)   ! Matriz resultado
    INTEGER, INTENT(IN) :: sizeX            ! Tamaño de renglones
    INTEGER, INTENT(IN) :: sizeY            ! Tamaño de columnas
    INTEGER :: i                            ! Indice del renglon
    INTEGER :: j                            ! Indice de la columna
    INTEGER :: k                            ! Indice de la multiplicacion
    REAL(long) :: result                    ! Resultado de la multiplicacion

    do i = 1, sizeX, 1
        do j = 1, sizeY, 1

            result = 0.0
            do k = 1, sizeY, 1
                result = result + matrixA((i - 1) * sizeY + k) * matrixB((k - 1) * sizeY + j)
            end do
            matrixC((i - 1) * sizeY + j) = result
        
        end do
    end do

end subroutine Multiply

! Suma todos los elementos de una matriz y regresa el resultado
subroutine AddMatrix(matrix, Nx, Ny, result)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: matrix(:)     ! Matriz resultado
    INTEGER, INTENT(IN) :: Nx               ! Tamaño de renglones
    INTEGER, INTENT(IN) :: Ny               ! Tamaño de columnas
    REAL(long), INTENT(OUT) :: result       ! Resultado de la suma
    INTEGER :: i                            ! Indice del renglon
    INTEGER :: j                            ! Indice de la columna

    result = 0.0_long

    do i = 1, Nx, 1
        do j = 1, Ny, 1
            result = result + matrix((i - 1) * Nx + j + 1)
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
