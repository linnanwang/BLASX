/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHER2 "

void cblas_zher2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int ldA)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (ldA < MAX(1,N))                                    info = 7;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zher2_p == NULL) blasx_init_cblas_func(&cblas_zher2_p, "cblas_zher2");
    Blasx_Debug_Output("Calling cblas_zher2\n ");
    (*cblas_zher2_p)(Order,Uplo,N,alpha,X,incX,Y,incY,A,ldA);
}

/* f77 interface */
void zher2_(char *uplo, int *n, double *alpha,
            double *X, int *incx, double *Y, int *incy, double *A, int *lda)
{
    enum CBLAS_UPLO Uplo;
    printf("uplo:%c\n",*uplo);
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zher2_ interface\n");
    cblas_zher2(CblasColMajor, Uplo, *n,
                alpha, X, *incx,
                Y, *incy, A, *lda);
}