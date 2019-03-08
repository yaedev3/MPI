/* File:    trap.c
 * Purpose: Calculate definite integral using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -o trap trap.c
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * IPP:     Section 3.2.1 (pp. 94 and ff.) and 5.2 (p. 216)
 */

#include <stdio.h>

double f(double x); /* Function we're integrating */
double Trap(double a, double b, int n, double h);

int main(void)
{
    double integral; /* Store result in integral */
    double a;        /* Left endpoints */
    double b;        /* Right endpoints */
    double h;        /* Height of trapezoids */
    int n;           /* Number of trapezoids */

    a = 0.0;
    b = 3.0;
    n = 1024;

    h = (b - a) / n;
    integral = Trap(a, b, n, h);

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15le\n", a, b, integral);

    return 0;
} /* main */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap(double a, double b, int n, double h)
{
    double integral;
    int k;

    integral = (a * a + b * b) / 2.0;

    for (k = 1; k <= n - 1; k++)
    {
        integral += (a + k * h) * (a + k * h);
//        printf("Calculado %lf total %lf\n", (a + k * h) * (a + k * h), integral);
    }

    integral = integral * h;

    return integral;
} /* Trap */
