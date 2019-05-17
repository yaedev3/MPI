MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

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


program main
    use precision
    use time_ftcs
    use iso_fortran_env
    implicit none
    include 'mpif.h'
    
    ! Calculate local integral
    INTERFACE 

       SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)
            use precision
            use iso_fortran_env
            IMPLICIT NONE
            REAL(long), INTENT(IN) :: left_endpt
            REAL(long), INTENT(IN) :: right_endpt
            REAL(long), INTENT(IN) :: base_len
            INTEGER, INTENT(IN) :: trap_count
            REAL(long), INTENT(OUT) :: estimate
       END SUBROUTINE Trap

       subroutine OpenFile(a, b, n)
            use precision
            use iso_fortran_env
            implicit none
            REAL(long), INTENT(OUT) :: a
            REAL(long), INTENT(OUT) :: b
            INTEGER, INTENT(OUT) :: n
        end subroutine OpenFile
    END INTERFACE
    
    ! Declaration of variables
    INTEGER :: my_rank 
    INTEGER :: comm_sz
    INTEGER :: n
    INTEGER :: local_n
    INTEGER :: ierror
    INTEGER :: status(MPI_STATUS_SIZE)
    INTEGER :: source
    REAL(long) :: a
    REAL(long) :: b
    REAL(long) :: h
    REAL(long) :: local_a
    REAL(long) :: local_b
    REAL(long) :: local_int
    REAL(long) :: total_int
    REAL(long) :: estimation                        ! Estimacion del calculo
    REAL(long) :: error                             ! Error encontrado
    REAL(long) :: elapsed                           ! Tiempo que tomo ejecutarse el programa
    REAL(long) :: start_clock                       ! Tiempo de inicio
    REAL(long) :: stop_clock                        ! Tiempo de termino
    CHARACTER(len=38) :: start                      ! Hora de inicio
    CHARACTER(len=38) :: end                        ! Hora de termino
    
    ! Let the system do what it needs to start up MPI
    call MPI_INIT(ierror)
    ! Get my process rank
    call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierror)
    ! Find out how many processes are being used
    call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_sz, ierror)
    
    if (my_rank == 0) then

        ! Establece la hora de inicio
        start = timestamp()
        call CPU_TIME(start_clock)

         ! Assignment
        CALL OpenFile(a, b, n)
        
    end if

    call MPI_BCAST(a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, ierror)
    call MPI_BCAST(b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, ierror)
    call MPI_BCAST(n, 1, MPI_LONG, 0, MPI_COMM_WORLD, ierror)

    h = (b - a) / n         ! h is the same for all processes 
    local_n = n / comm_sz   ! So is the number of trapezoids 
    
    ! Length of each process' interval of
    ! integration = local_n * h.  So my interval
    ! starts at:
    local_a = a + my_rank * local_n * h
    local_b = local_a + local_n * h
    call Trap (local_a, local_b, local_n, h, local_int)

    call MPI_REDUCE(local_int, total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, ierror)

    ! Print the result
    if (my_rank == 0) then
        
         ! Calculo estimado con la formula a^2*N^3
        estimation = 9.0_long

        ! Calcula el % de error
        error = DABS(total_int - estimation) / estimation * 100.0_long

        end = timestamp()
        call CPU_TIME(stop_clock)
        
        ! Calcula el tiempo que tomo ejecutar el programa
        elapsed =  stop_clock - start_clock

        ! Guarda el resultado en un archivo
        call SaveFile(start, end, error, elapsed, N, comm_sz)

    end if    
    
    call MPI_FINALIZE(ierror)
    
end program main
    
!---------------------------------------------------------------------
!
!  Subroutine for estimating a definite integral 
!  using the trapezoidal rule
! Input args:   left_endpt
!               right_endpt
!               trap_count 
!               base_len
! Output val:   Trapezoidal rule estimate of integral from
!               left_endpt to right_endpt using trap_count
!               trapezoids
!
!---------------------------------------------------------------------
SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)
    use precision
    use iso_fortran_env
    IMPLICIT NONE
    REAL(long), INTENT(IN) :: left_endpt
    REAL(long), INTENT(IN) :: right_endpt
    REAL(long), INTENT(IN) :: base_len
    INTEGER, INTENT(IN) :: trap_count
    REAL(long), INTENT(OUT) :: estimate
    REAL(long) :: x
    INTEGER :: i
    
    estimate = ( left_endpt ** 2 + right_endpt ** 2 ) / 2_long;
    
    do i = 1, trap_count - 1, 1
    
        x = left_endpt + i * base_len
        estimate = estimate + x ** 2
    
    end do
    
    estimate = estimate * base_len
    
END SUBROUTINE Trap

subroutine OpenFile(a, b, n)
    use precision
    use iso_fortran_env
    implicit none
    REAL(long), INTENT(OUT) :: a
    REAL(long), INTENT(OUT) :: b
    INTEGER, INTENT(OUT) :: n
    INTEGER :: stat
    CHARACTER(len=30) :: input_file

    input_file = 'parameters.dat'

    OPEN(UNIT=1,file=input_file,ACTION="READ",IOSTAT=stat,STATUS='OLD')
    !a = 0.0_long
    !b = 3.0_long
    READ(1, *) a
    READ(1, *) b
    READ(1, *) n

    CLOSE(1)

end subroutine OpenFile


! Crea un archivo de salida con la hora de inicio, de termino y el tiempo que tomo correr el programa
! Asi como el porcentaje de error
subroutine SaveFile(start, end, error, elapsed, N, process)
    use precision
    CHARACTER(len=38), INTENT(IN) :: start
    CHARACTER(len=38), INTENT(IN) :: end
    REAL(long), INTENT(IN) :: error
    REAL(long), INTENT(IN) :: elapsed
    INTEGER, INTENT(IN) :: N
    INTEGER, INTENT(IN) :: process
    CHARACTER(len = 84) :: file_name
    CHARACTER(len = 15) :: Nstring
    CHARACTER(len = 1) :: processString

    write(Nstring, "(I15)") N 
    write(processString, "(I1)") process

    file_name = 'mpi-f90-'//processString//'-'//Nstring//'-'//trim(end)//'.txt'
    
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
