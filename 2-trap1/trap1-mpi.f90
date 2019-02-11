program trap1
    implicit none
    include 'mpif.h'
    
    ! Calculate local integral
    INTERFACE 
       SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)
        REAL, INTENT(IN) :: left_endpt, right_endpt, base_len
        INTEGER, INTENT(IN) :: trap_count
        REAL, INTENT(OUT) :: estimate
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
    REAL :: a
    REAL :: b
    REAL :: h
    REAL :: local_a
    REAL :: local_b
    REAL :: local_int
    REAL :: total_int
    
    ! Let the system do what it needs to start up MPI
    call MPI_INIT(ierror)
    ! Get my process rank
    call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierror)
    ! Find out how many processes are being used
    call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_sz, ierror)
    
    ! Assignment
    n = 1024;
    a = 0.0;
    b = 14.0;
    h = (b - a) / n;       ! h is the same for all processes 
    local_n = n / comm_sz; ! So is the number of trapezoids 
    
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
            total_int = total_int + local_int;
        end do
    else
        call MPI_SEND(local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, ierror)
    end if

    ! Print the result
    if (my_rank == 0) then
        write (*, *) 'With n = ' , n, ' trapezoids, our estimate'
        write (*, *) 'of the integral from', a, ' to ' , b, ' = ', local_int
    end if    
    
    call MPI_FINALIZE(ierror)
    
end program trap1
    
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
    
    IMPLICIT NONE
    REAL, INTENT(IN) :: left_endpt
    REAL, INTENT(IN) :: right_endpt
    REAL, INTENT(IN) :: base_len
    INTEGER, INTENT(IN) :: trap_count
    REAL, INTENT(OUT) :: estimate
    REAL :: x
    INTEGER :: i
    
    estimate = ( left_endpt * left_endpt + right_endpt * right_endpt ) / 2.;
    
    do i = 1, trap_count, 1
    
        x = left_endpt + i * base_len
        estimate = estimate + ( x * x )
    
    end do
    
    estimate = estimate * base_len
    
END SUBROUTINE Trap