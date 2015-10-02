#include "blasx.h"

float cblas_snrm2(const int N, const float *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_snrm2_p == NULL) blasx_init_cblas_func(&cblas_snrm2_p, "cblas_snrm2");
    Blasx_Debug_Output("Calling cblas_snrm2 interface\n");
    return (*cblas_snrm2_p)(N,X,incX);
}

float snrm2_(int *n, float *X, int *incx)
{
    Blasx_Debug_Output("Calling snrm2_ interface\n");
    return cblas_snrm2(*n,X,*incx);
}


double cblas_dnrm2(const int N, const double *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dnrm2_p == NULL) blasx_init_cblas_func(&cblas_dnrm2_p, "cblas_dnrm2");
    Blasx_Debug_Output("Calling cblas_dnrm2 interface\n");
    return (*cblas_dnrm2_p)(N,X,incX);
}

double dnrm2_(int *n, double *X, int *incx)
{
    Blasx_Debug_Output("Calling dnrm2_ interface\n");
    return cblas_dnrm2(*n,X,*incx);
}


float cblas_scnrm2(const int N, const void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_scnrm2_p == NULL) blasx_init_cblas_func(&cblas_scnrm2_p, "cblas_scnrm2");
    Blasx_Debug_Output("Calling cblas_scnrm2 interface\n");
    return (*cblas_scnrm2_p)(N,X,incX);
}

float scnrm2_(int *n, float *X, int *incx)
{
    Blasx_Debug_Output("Calling scnrm2_ interface\n");
    return cblas_scnrm2(*n,X,*incx);
}

double cblas_dznrm2(const int N, const void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dznrm2_p == NULL) blasx_init_cblas_func(&cblas_dznrm2_p, "cblas_dznrm2");
    Blasx_Debug_Output("Calling cblas_dznrm2 interface\n");
    return (*cblas_dznrm2_p)(N,X,incX);
}

double dznrm2_(int *n, float *X, int *incx)
{
    Blasx_Debug_Output("Calling dznrm2 interface\n");
    return cblas_dznrm2(*n,X,*incx);
}







