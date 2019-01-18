program trap1
implicit none

INTERFACE 
   SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)
    REAL, INTENT(IN) :: left_endpt, right_endpt, base_len
    INTEGER, INTENT(IN) :: trap_count
    REAL, INTENT(OUT) :: estimate
   END SUBROUTINE Trap
END INTERFACE

integer :: n
real :: a, b, h, local_int, local_a, local_b

n = 1024
a = 0.0
b = 14.0
h = (b - a) / n; ! h is the same for all processes 

local_a = a + h
local_b = local_a + h

call Trap (local_a, local_b, n, h, local_int)

write (*, *) 'With n = ' , n, ' trapezoids, our estimate'
write (*, *) 'of the integral from', a, ' to ' , b, ' = ', local_int

end program trap1

!---------------------------------------------------------------------
!
!  Subroutine to 
!
!---------------------------------------------------------------------
SUBROUTINE Trap(left_endpt, right_endpt, trap_count, base_len, estimate)

IMPLICIT NONE
REAL, INTENT(IN) :: left_endpt, right_endpt, base_len
INTEGER, INTENT(IN) :: trap_count
REAL, INTENT(OUT) :: estimate
REAL :: x
INTEGER :: i

estimate = ( left_endpt * left_endpt + right_endpt * right_endpt ) / 2.;

do i = 1, trap_count, 1

    x = left_endpt + i * base_len
    estimate = estimate + ( x * x )

end do

estimate = estimate * base_len / 100

END SUBROUTINE Trap