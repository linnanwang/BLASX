/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "SSBMV "

//SBMV
void cblas_ssbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY)

{
    cublasFillMode_t uplo;
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
    if (cblas_ssbmv_p == NULL) blasx_init_cblas_func(&cblas_ssbmv_p, "cblas_ssbmv");
    Blasx_Debug_Output("Calling cblas_ssbmv\n ");
    (*cblas_ssbmv_p)(Order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void ssbmv_(char *uplo, int *n, int *k, float *alpha, float *A, int *lda,
            float *X, int *incx, float *beta, float *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling ssbmv_ interface\n");
    cblas_ssbmv(CblasColMajor, Uplo,
                *n, *k, *alpha, A,
                *lda, X, *incx,
                *beta, Y, *incy);
}