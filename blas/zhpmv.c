/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHPMV "

void cblas_zhpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *Ap,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
    Blasx_Debug_Output("Calling cblas_zhpmv interface\n");
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 6;
    else if (incY == 0)                                         info = 9;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zhpmv_p == NULL) blasx_init_cblas_func(&cblas_zhpmv_p, "cblas_zhpmv");
    Blasx_Debug_Output("Calling cblas_zhpmv\n ");
    (*cblas_zhpmv_p)(Order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);
}

/* f77 interface */
void zhpmv_(char *uplo, int *n, double *alpha, double *Ap,
            double *X, int *incx, double *beta, double *Y, int *incy)
{
    
    Blasx_Debug_Output("Calling zhpmv_ interface\n");
    Blasx_Debug_Output("uplo:%c\n",*uplo);
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_zhpmv(CblasColMajor, Uplo,
                *n, alpha, Ap,
                X, *incx,
                beta, Y, *incy);
}