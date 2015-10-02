/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHPR2 "

void cblas_zhpr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *Ap)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (incY == 0)                                         info = 7;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zhpr2_p == NULL) blasx_init_cblas_func(&cblas_zhpr2_p, "cblas_zhpr2");
    Blasx_Debug_Output("Calling cblas_zhpr2\n ");
    (*cblas_zhpr2_p)(Order,Uplo,N,alpha,X,incX,Y,incY,Ap);
}

/* f77 interface */
void zhpr2_(char *uplo, int *n, double *alpha,
            double *X, int *incx, double *Y, int *incy, double *Ap)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zhpr2_ interface\n");
    cblas_zhpr2(CblasColMajor, Uplo, *n,
                alpha, X, *incx,
                Y, *incy, Ap);
}