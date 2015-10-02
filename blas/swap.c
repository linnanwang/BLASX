#include <blasx.h>

void cblas_sswap(const int N, float *X, const int incX, float *Y, const int incY)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sswap_p == NULL) blasx_init_cblas_func(&cblas_sswap_p, "cblas_sswap");
    Blasx_Debug_Output("Calling cblas_sswap\n ");
    (*cblas_sswap_p)(N,X,incX,Y,incY);
}
void sswap_(int *n, float *X, int *incx, float *Y, int *incy)
{
    Blasx_Debug_Output("Calling sswap_\n ");
    cblas_sswap(*n,X,*incx,Y,*incy);
}

void cblas_dswap(const int N, double *X, const int incX, double *Y, const int incY)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dswap_p == NULL) blasx_init_cblas_func(&cblas_dswap_p, "cblas_dswap");
    Blasx_Debug_Output("Calling cblas_dswap\n ");
    (*cblas_dswap_p)(N,X,incX,Y,incY);
}
void dswap_(int *n, double *X,int *incx, double *Y, int *incy)
{
    Blasx_Debug_Output("Calling dswap_\n ");
    cblas_dswap(*n,X,*incx,Y,*incy);
}

void cblas_cswap(const int N, void *X, const int incX, void *Y, const int incY)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_cswap_p == NULL) blasx_init_cblas_func(&cblas_cswap_p, "cblas_cswap");
    Blasx_Debug_Output("Calling cblas_cswap\n ");
    (*cblas_cswap_p)(N,X,incX,Y,incY);
}
void cswap_(int *n, float *X,int *incx, float *Y, int *incy)
{
    Blasx_Debug_Output("Calling cswap_\n ");
    cblas_cswap(*n,X,*incx,Y,*incy);
}

void cblas_zswap(const int N, void *X, const int incX, void *Y, const int incY)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zswap_p == NULL) blasx_init_cblas_func(&cblas_zswap_p, "cblas_zswap");
    Blasx_Debug_Output("Calling cblas_zswap\n ");
    (*cblas_zswap_p)(N,X,incX,Y,incY);
}
void zswap_(int *n, double *X,int *incx, double *Y,int *incy)
{
    Blasx_Debug_Output("Calling zswap_\n ");
    cblas_zswap(*n,X,*incx,Y,*incy);
}