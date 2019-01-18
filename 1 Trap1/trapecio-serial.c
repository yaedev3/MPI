#include <stdio.h>

/* Calculate local integral  */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len);
/* Function we're integrating */
double f(double x);

void main()
{
    int n;
    double a, b, h, local_int, local_a, local_b;

    n = 1024;
    a = 0.0;
    b = 14.0;
    h = (b - a) / n; /* h is the same for all processes */

    local_a = a + h;
    local_b = local_a + h;
    local_int = Trap(local_a, local_b, n, h);

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15e\n", a, b, local_int);
}

/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral 
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count 
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len)
{
    double estimate, x;
    int i;

    estimate = (f(left_endpt) + f(right_endpt)) / 2.0;
    for (i = 1; i <= trap_count - 1; i++)
    {
        x = left_endpt + i * base_len;
        estimate += f(x);
    }
    estimate = estimate * base_len;

    return estimate;
} /*  Trap  */

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x)
{
    return x * x;
} /* f */
