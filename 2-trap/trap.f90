MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15,307) !(6,37) for real(float), (15,307) for double
END MODULE precision

program main
    use precision
    implicit none

    ! Calculate local integral
    INTERFACE 
    SUBROUTINE Trap(a, b, n, h, integral)
        use precision
        implicit none
        REAL(long), INTENT(IN) :: a
        REAL(long), INTENT(IN) :: b
        INTEGER, INTENT(IN) :: n
        REAL(long), INTENT(IN) :: h
        REAL(long), INTENT(OUT) :: integral
    END SUBROUTINE Trap
    END INTERFACE

    ! Declaration of variables
    REAL(long) :: integral
    REAL(long) :: a
    REAL(long) :: b
    REAL(long) :: h
    INTEGER :: n

    ! Assignment
    a = 0.0_long
    b = 3.0_long
    n = 1024

    h = (b - a) / n;
    CALL Trap (a, b, n, h, integral)

    write (*, *) 'With n = ' , n, ' trapezoids, our estimate'
    write (*, *) 'of the integral from', a, ' to ' , b, ' = ', integral

end program main

! ------------------------------------------------------------------
! Function:    Trap
! Purpose:     Estimate integral from a to b of f using trap rule and
!              n trapezoids
! Input args:  a, b, n, h
! Return val:  Estimate of the integral 
! ------------------------------------------------------------------
SUBROUTINE Trap(a, b, n, h, integral)
    use precision
    implicit none
    REAL(long), INTENT(IN) :: a
    REAL(long), INTENT(IN) :: b
    INTEGER, INTENT(IN) :: n
    REAL(long), INTENT(IN) :: h
    REAL(long), INTENT(OUT) :: integral
    INTEGER :: k

    integral = ( a**2 + b**2 ) / 2_long

    do k = 1, n - 1, 1
        integral = integral + (a + k * h) ** 2
    end do

    integral = integral * h

END SUBROUTINE Trap