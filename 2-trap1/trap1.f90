program trap1
    implicit none

    ! Calculate local integral
    INTERFACE 
    SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)
        REAL, INTENT(IN) :: left_endpt, right_endpt, base_len
        INTEGER, INTENT(IN) :: trap_count
        REAL, INTENT(OUT) :: estimate
    END SUBROUTINE Trap
    END INTERFACE

    ! Declaration of variables
    INTEGER :: n
    REAL :: a
    REAL :: b
    REAL :: h
    REAL :: local_int
    REAL :: local_a
    REAL :: local_b

    ! Assignment
    n = 1024
    a = 0.0
    b = 14.0
    h = (b - a) / n;

    local_a = a + h
    local_b = local_a + h

    CALL Trap (local_a, local_b, n, h, local_int)

    write (*, *) 'With n = ' , n, ' trapezoids, our estimate'
    write (*, *) 'of the integral from', a, ' to ' , b, ' = ', local_int

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