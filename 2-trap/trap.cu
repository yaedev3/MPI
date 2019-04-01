#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

int main(int argc, char *argv[])
{
    double integral; /* Store result in integral */
    double a;        /* Left endpoints */
    double b;        /* Right endpoints */
    double h;        /* Height of trapezoids */
    int n;           /* Number of trapezoids */

    OpenFile(&a, &b, &n);

    h = (b - a) / n;
    integral = Trap(a, b, n, h);

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15le\n", a, b, integral);

    return 0;
}