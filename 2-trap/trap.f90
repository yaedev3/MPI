MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15, 307) !(6,37) for real(float), (15,307) for double
END MODULE precision

program main
    use precision
    implicit none

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

    subroutine OpenFile(a, b, n)
        use precision
        implicit none
        REAL(long), INTENT(OUT) :: a
        REAL(long), INTENT(OUT) :: b
        INTEGER, INTENT(OUT) :: n
    end subroutine OpenFile

    END INTERFACE

    ! Declaration of variables
    REAL(long) :: integral
    REAL(long) :: a
    REAL(long) :: b
    REAL(long) :: h
    INTEGER :: n

    ! Assignment
    CALL OpenFile(a, b, n)

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

subroutine OpenFile(a, b, n)
    use precision
    implicit none
    REAL(long), INTENT(OUT) :: a
    REAL(long), INTENT(OUT) :: b
    INTEGER, INTENT(OUT) :: n
    INTEGER :: stat
    CHARACTER(len=30) :: input_file

    input_file = 'parameters.dat'

    OPEN(UNIT=1,file=input_file,ACTION="READ",IOSTAT=stat,STATUS='OLD')
    if (stat .ne. 0) then
        a = 0.0_long
        b = 3.0_long
        n = 1024
        write(*, *) 'No se encontro el archivo \"parameters.dat\" se usaran parametros por defecto.'
    else 
        READ(1, *) a
        READ(1, *) b
        READ(1,*) n
    end if

    CLOSE(1)

end subroutine OpenFile