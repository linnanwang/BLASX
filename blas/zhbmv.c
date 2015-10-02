/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHBMV "

void cblas_zhbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const void *alpha, const void *A,
                 const int ldA, const void *X, const int incX,
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
    else if (ldA < (K+1))                                       info = 6;
    else if (incX == 0)                                         info = 8;
    else if (incY == 0)                                         info = 11;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zhbmv_p == NULL) blasx_init_cblas_func(&cblas_zhbmv_p, "cblas_zhbmv");
    Blasx_Debug_Output("Calling cblas_zhbmv\n ");
    (*cblas_zhbmv_p)(Order,Uplo,N,K,alpha,A,ldA,X,incX,beta,Y,incY);
}

/* f77 interface */
void zhbmv_(char *uplo, int *n, int *k, double *alpha, double *A, int *lda,
            double *X, int *incx, double *beta, double *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zhbmv_ interface\n");
    cblas_zhbmv(CblasColMajor,Uplo,
                *n, *k, alpha, A,
                *lda, X, *incx,
                beta,Y, *incy);
}
