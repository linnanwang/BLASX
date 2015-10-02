#include "blasx.h"

void cblas_scopy(const int N, const float *X, const int incX,
                      float *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_scopy_p == NULL) blasx_init_cblas_func(&cblas_scopy_p, "cblas_scopy");
    Blasx_Debug_Output("Calling cblas_scopy interface\n");
    (*cblas_scopy_p)(N,X,incX,Y,incY);
}
void scopy_(int *n, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling scopy_ interface\n");
    cblas_scopy(*n,X,*incx,Y,*incy);
}


void cblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dcopy_p == NULL) blasx_init_cblas_func(&cblas_dcopy_p, "cblas_dcopy");
    Blasx_Debug_Output("Calling cblas_dcopy interface\n");
    (*cblas_dcopy_p)(N,X,incX,Y,incY);

}
void dcopy_(int *n, double *X, int *incx, double *Y, int *incy){
    Blasx_Debug_Output("Calling dcopy_ interface\n");
    cblas_dcopy(*n,X,*incx,Y,*incy);
}


void cblas_ccopy(const int N, const void *X, const int incX,
                 void *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_ccopy_p == NULL) blasx_init_cblas_func(&cblas_ccopy_p, "cblas_ccopy");
    Blasx_Debug_Output("Calling cblas_ccopy interface\n");
    (*cblas_ccopy_p)(N,X,incX,Y,incY);
}
void ccopy_(int *n, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling ccopy_ interface\n");
    cblas_ccopy(*n,X,*incx,Y,*incy);
}


void cblas_zcopy(const int N, const void *X, const int incX,
                 void *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zcopy_p == NULL) blasx_init_cblas_func(&cblas_zcopy_p, "cblas_zcopy");
    Blasx_Debug_Output("Calling cblas_zcopy interface\n");
    (*cblas_zcopy_p)(N,X,incX,Y,incY);
}
void zcopy_(int *n, double *X, int *incx, double *Y, int *incy){
    Blasx_Debug_Output("Calling zcopy_ interface\n");
    cblas_zcopy(*n,X,*incx,Y,*incy);
}