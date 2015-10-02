/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHBMV "

void cblas_chbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
    cublasOperation_t transa;
    cublasFillMode_t uplo;
    cublasStatus_t status;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (K < 0)                                             info = 3;
    else if (lda < (K+1))                                       info = 6;
    else if (incX == 0)                                         info = 8;
    else if (incY == 0)                                         info = 11;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_chbmv_p == NULL) blasx_init_cblas_func(&cblas_chbmv_p, "cblas_chbmv");
    Blasx_Debug_Output("Calling cblas_chbmv\n ");
    (*cblas_chbmv_p)(Order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void chbmv_(char *uplo, int *n, int *k, float *alpha, float *A, int *lda,
            float *X, int *incx, float *beta, float *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling chbmv_ interface\n");
    cblas_chbmv(CblasColMajor,Uplo,
                *n, *k, alpha, A,
                *lda, X, *incx,
                beta,Y, *incy);
}
