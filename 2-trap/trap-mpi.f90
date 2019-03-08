MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

program main
    use precision
    implicit none
    include 'mpif.h'
    
    ! Calculate local integral
    INTERFACE 
       SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)
       use precision
       IMPLICIT NONE
       REAL(long), INTENT(IN) :: left_endpt
       REAL(long), INTENT(IN) :: right_endpt
       REAL(long), INTENT(IN) :: base_len
       INTEGER, INTENT(IN) :: trap_count
       REAL(long), INTENT(OUT) :: estimate
       END SUBROUTINE Trap
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
    
    ! Let the system do what it needs to start up MPI
    call MPI_INIT(ierror)
    ! Get my process rank
    call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierror)
    ! Find out how many processes are being used
    call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_sz, ierror)
    
    ! Assignment
    n = 1024_long
    a = 0.0_long
    b = 3.0_long

    h = (b - a) / n         ! h is the same for all processes 
    local_n = n / comm_sz   ! So is the number of trapezoids 
    
    ! Length of each process' interval of
    ! integration = local_n * h.  So my interval
    ! starts at:
    local_a = a + my_rank * local_n * h
    local_b = local_a + local_n * h
    call Trap (local_a, local_b, local_n, h, local_int)

    ! Add up the integrals calculated by each process
    if(my_rank == 0) then
        total_int = local_int

        do source = 1, comm_sz - 1, 1

            call MPI_RECV(local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, status, ierror)
            total_int = total_int + local_int

        end do
    else
        call MPI_SEND(local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, ierror)
    end if

    ! Print the result
    if (my_rank == 0) then
        write (*, *) 'With n = ' , n, ' trapezoids, our estimate'
        write (*, *) 'of the integral from', a, ' to ' , b, ' = ', total_int
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