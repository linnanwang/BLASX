#include <blasx_globalpointers.h>
#include <blasx_tile_resource.h>
#include <math.h>

float cblas_sasum(const int N, const float *X, const int incX){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sasum_p == NULL) blasx_init_cblas_func(&cblas_sasum_p, "cblas_sasum");
    Blasx_Debug_Output("Calling cblas_sasum\n ");
    return (*cblas_sasum_p)(N,X,incX);
}

float sasum_(int *n, float *X, int *incx)
{
    Blasx_Debug_Output("Calling sasum_ interface\n");
    return cblas_sasum(*n,X,*incx);
}

double cblas_dasum(const int N, const double *X, const int incX){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dasum_p == NULL) blasx_init_cblas_func(&cblas_dasum_p, "cblas_dasum");
    Blasx_Debug_Output("Calling cblas_dasum\n ");
    return (*cblas_dasum_p)(N,X,incX);
}

double dasum_(int *n, double *X, int *incx){
    Blasx_Debug_Output("Calling dasum_ interface\n");
    return cblas_dasum(*n,X,*incx);
}


float cblas_scasum(const int N, const void *X, const int incX){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_scasum_p == NULL) blasx_init_cblas_func(&cblas_scasum_p, "cblas_scasum");
    Blasx_Debug_Output("Calling cblas_scasum\n ");
    return (*cblas_scasum_p)(N,X,incX);
}

float scasum_(int *n, float *X, int *incx){
    Blasx_Debug_Output("Calling scasum_ interface\n");
    return cblas_scasum(*n,X,*incx);
}

double cblas_dzasum(const int N, const void *X, const int incX){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dzasum_p == NULL) blasx_init_cblas_func(&cblas_dzasum_p, "cblas_dzasum");
    Blasx_Debug_Output("Calling cblas_dzasum\n ");
    return (*cblas_dzasum_p)(N,X,incX);
}

double dzasum_(int *n, double *X, int *incx){
    Blasx_Debug_Output("Calling dzasum_ interface\n");
    return cblas_dzasum(*n,X,*incx);
}




