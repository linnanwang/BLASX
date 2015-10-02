#include "blasx.h"

void cblas_saxpy(const int N, const float alpha, const float *X,
                   const int incX, float *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_saxpy_p == NULL) blasx_init_cblas_func(&cblas_saxpy_p, "cblas_saxpy");
    Blasx_Debug_Output("Calling cblas_saxpy interface\n");
    (*cblas_saxpy_p)(N,alpha,X,incX,Y,incY);
}

void saxpy_(int *n, float *alpha, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling saxpy_ interface\n");
    cblas_saxpy(*n,*alpha,X,*incx,Y,*incy);
}

void cblas_daxpy(const int N, const double alpha, const double *X,
                      const int incX, double *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_daxpy_p == NULL) blasx_init_cblas_func(&cblas_daxpy_p, "cblas_daxpy");
    Blasx_Debug_Output("Calling cblas_daxpy interface\n");
    (*cblas_daxpy_p)(N,alpha,X,incX,Y,incY);
}
void daxpy_(int *n, double *alpha, double *X, int *incx, double *Y, int *incy){
    Blasx_Debug_Output("Calling daxpy_ interface\n");
    cblas_daxpy(*n,*alpha,X,*incx,Y,*incy);
}

void cblas_caxpy(const int N, const void *alpha, const void *X,
                      const int incX, void *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_caxpy_p == NULL) blasx_init_cblas_func(&cblas_caxpy_p, "cblas_caxpy");
    Blasx_Debug_Output("Calling cblas_caxpy interface\n");
    (*cblas_caxpy_p)(N,alpha,X,incX,Y,incY);
}

void caxpy_(int *n, float *alpha, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling caxpy_ interface\n");
    cblas_caxpy(*n,alpha,X,*incx,Y,*incy);
}

void cblas_zaxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zaxpy_p == NULL) blasx_init_cblas_func(&cblas_zaxpy_p, "cblas_zaxpy");
    Blasx_Debug_Output("Calling cblas_zaxpy interface\n");
    (*cblas_zaxpy_p)(N,alpha,X,incX,Y,incY);
}

void zaxpy_(int *n, double *alpha, double *X, int *incx, double *Y, int *incy){
    Blasx_Debug_Output("Calling zaxpy_ interface\n");
    cblas_zaxpy(*n,alpha,X,*incx,Y,*incy);
}