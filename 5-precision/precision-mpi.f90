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

    INTEGER :: N
    REAL(long), POINTER, DIMENSION(:) :: matrixA
    REAL(long), POINTER, DIMENSION(:) :: matrixB
    REAL(long), POINTER, DIMENSION(:) :: matrixC
    REAL(long) :: result
    REAL(long) :: result_local
    REAL(long) :: estimation
    REAL(long) :: error
    INTEGER :: rank 
    INTEGER :: process
    INTEGER :: processSize
    INTEGER :: ierror
    INTEGER :: status(MPI_STATUS_SIZE)
    INTEGER :: waste
    INTEGER :: n_local
    INTEGER :: i
 
    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, process, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

    ! Asigna el tamaño de la matriz
    call OpenFile(N)
    
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

    ! Comparte la informacion de la matriz B con los demas procesos.
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

        ! Calculo estimado con la formula a^2*N^3.
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

        ! Inicializa las matrices A y C con los tamaños correspondientes a cada proceso.
        allocate(matrixA(N * processSize))
        allocate(matrixC(N * processSize))

        ! Recibe la matriz A del proceso 0
        call MPI_RECV(matrixA, processSize * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, status, ierror)

        ! Hace la multiplicacion de matrices.
        call Multiply(matrixA, matrixB, matrixC, processSize, N)

        ! Hace la suma de todos los elementos de la matriz C
        call AddMatrix(MatrixC, processSize, N, result_local)

        ! Envia el resultado de la multiplicacion al proceso 0.
        call MPI_SEND(result_local, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, ierror)

        ! Libera la memoria de la matriz C
        deallocate(matrixC)

    end if

    ! Libera la memoria de las matrices A y B
    deallocate(matrixA)
    deallocate(matrixB)

    call MPI_FINALIZE(ierror)

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

subroutine Multiply(MatrixA, MatrixB, MatrixC, sizeX, sizeY)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: MatrixA(:)
    REAL(long), INTENT(IN) :: MatrixB(:)
    REAL(long), INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: sizeX
    INTEGER, INTENT(IN) :: sizeY
    INTEGER :: i
    INTEGER :: j
    INTEGER :: k
    REAL(long) :: result

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

subroutine AddMatrix(matrix, Nx, Ny, result)
    use precision
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: matrix(:)
    INTEGER, INTENT(IN) :: Nx
    INTEGER, INTENT(IN) :: Ny
    REAL(long), INTENT(OUT) :: result
    INTEGER :: i
    INTEGER :: j

    result = 0.0_long

    do i = 1, Nx, 1
        do j = 1, Ny, 1
            result = result + matrix((i - 1) * Nx + j + 1)
        end do
    end do

end subroutine AddMatrix

subroutine OpenFile(N)
    implicit none
    INTEGER, INTENT(OUT) :: N
    CHARACTER(len=30) :: input_file

    input_file = 'N.dat'

    OPEN(UNIT=1,file=input_file,ACTION="READ",STATUS='OLD')
    READ(1,*) N
    CLOSE(1)

end subroutine OpenFile