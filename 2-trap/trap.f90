MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(15, 307) !(6,37) for real(float), (15,307) for double
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

    ! Assignment
    call OpenFile(a, b, n)

    h = (b - a) / n;
    call Trap (a, b, n, h, integral)

    ! Estimacion del resultado
    estimation = 9.0_long

    ! Calcula el % de error
    error = DABS(integral - estimation) / estimation * 100.0_long

    ! Establece la hora en que termino de calcular
    end = timestamp()
    call CPU_TIME(stop_clock)

    ! Calcula el tiempo que tomo ejecutar el programa
    elapsed =  stop_clock - start_clock

    ! Guarda el resultado en un archivo
    call SaveFile(start, end, error, elapsed, N)

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
    READ(1, *) a
    READ(1, *) b
    READ(1,*) n
 
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
    CHARACTER(len = 8) :: Nstring

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