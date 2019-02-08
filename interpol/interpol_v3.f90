MODULE precision
  INTEGER ,PARAMETER:: long=SELECTED_REAL_KIND(8,307) !8 for real(float), 16 for double
END MODULE precision

MODULE parameters
  USE precision
  IMPLICIT NONE
  REAL(long), PARAMETER :: Pi            =   ATAN(1.0)*4.0
END MODULE parameters

PROGRAM MAIN
  USE precision
  USE parameters
  IMPLICIT NONE

  INTEGER :: i,dim_real,j
  INTEGER, PARAMETER :: dim_max = 100000
  REAL(long), DIMENSION(1:dim_max) :: XVEC,YVEC
  CHARACTER(len=30) :: input_file
  REAL(long) :: ratio, alpha 
  
  input_file = 'D_lj_red_file.dat' 
  ratio = 1.0E-4 ! larger than 1.3E-7
  
  dim_real = 0
  OPEN(UNIT=1,file=input_file,ACTION="READ",STATUS='OLD')
  DO j=1,dim_max
     READ(1,*,END=10) XVEC(j),YVEC(j)
     dim_real = dim_real + 1
  END DO
  10 PRINT *,'End of file',dim_real
  PRINT *,XVEC(1),YVEC(1)
  PRINT *,XVEC(dim_real),YVEC(dim_real)
  CLOSE(1) 

  CALL INTERPOLATION(dim_max,dim_real,XVEC,YVEC,ratio,alpha)

  PRINT *,'ratio',ratio
  PRINT *,'alpha',alpha
  
  END PROGRAM

  SUBROUTINE INTERPOLATION(dim_max_vec,dim_real_vec,X,Y,ratio,result)
    USE precision
    IMPLICIT NONE
    INTEGER,INTENT(IN) :: dim_max_vec
    INTEGER,INTENT(IN) :: dim_real_vec
    REAL(long),DIMENSION(1:dim_max_vec),INTENT(IN) :: X 
    REAL(long),DIMENSION(1:dim_max_vec),INTENT(IN) :: Y
    REAL(long),INTENT(IN) :: ratio
    REAL(long),INTENT(OUT) :: result
    INTEGER :: i,index
    REAL(long) :: slope,alpha
    LOGICAL :: CONTINUE = .FALSE.

    alpha = 0.0 
    
    DO i=1,dim_real_vec
       IF (Y(i)>=ratio) THEN
          index = i
          CONTINUE = .TRUE.
       END IF
    END DO

    IF (CONTINUE .EQV. .FALSE.) THEN
       PRINT *,'Cociente fuera de rango'
       STOP
    ELSE IF (index>1) THEN
    
    slope = (X(index)-X(index-1))/(Y(index)-Y(index-1))

    alpha = X(index-1) + slope*(ratio-Y(index-1))

    END IF 

    result = alpha
    
  END SUBROUTINE INTERPOLATION
    

