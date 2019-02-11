#include <stdio.h>
#include <stdlib.h>

typedef short bool;

#define true 1
#define false 0

double Interpolacion(int size, double *X, double *Y, double ratio);

void main()
{
    int dim_real;
    int dim_max;
    double *XVEC;
    double *YVEC;
    double ratio;
    double alpha;
    char *input_file;
    FILE *file;

    dim_max = 100000;
    dim_real = 0;
    input_file = "D_lj_red_file.dat";
    file = fopen(input_file, "r");
    ratio = 1.0E-4;
    XVEC = (double *)malloc(sizeof(double) * dim_max);
    YVEC = (double *)malloc(sizeof(double) * dim_max);

    if (file == NULL)
        printf("No se puede abrir el archivo.\n");
    else
    {
        while (!feof(file)) 
        {
            fscanf(file, "%lf %lf", &XVEC[dim_real], &YVEC[dim_real]);
            dim_real++;
        }

        fclose(file);
        printf("End of file\n");
        printf("%.15le %.15le\n", XVEC[0], YVEC[0]);
        printf("%.15le %.15le\n", XVEC[dim_real - 1], YVEC[dim_real - 1]);

        alpha = Interpolacion(dim_real, XVEC, YVEC, ratio);
        printf("ratio %.15le\n alpha %.17le\n", ratio, alpha);
    }

    free(XVEC);
    free(YVEC);
}

double Interpolacion(int size, double *X, double *Y, double ratio)
{
    int i;
    int index;
    double result;
    double slope;
    double alpha;
    bool _continue;

    alpha = 0.0;
    _continue = false;

    for (i = 0; i < size; i++)
        if (Y[i] >= ratio)
        {
            index = i;
            _continue = true;
        }

    if (_continue == false)
        printf("Cociente fuera del rango\n");
    else
    {
        slope = (X[index] - X[index - 1]) / (Y[index] - Y[index - 1]);
        alpha = X[index - 1] + slope * (ratio - Y[index - 1]);
    }

    result = alpha;

    return result;
}