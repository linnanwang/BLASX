/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "SSYMV "

void cblas_ssymv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (lda < MAX(1,N))                                    info = 5;
    else if (incX == 0)                                         info = 7;
    else if (incY == 0)                                         info = 10;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_ssymv_p == NULL) blasx_init_cblas_func(&cblas_ssymv_p, "cblas_ssymv");
    Blasx_Debug_Output("Calling cblas_ssymv\n ");
    (*cblas_ssymv_p)(Order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void ssymv_(char *uplo, int *n, float *alpha, float *A, int *lda,
            float *X, int *incx, float *beta, float *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling ssymv_ interface\n");
    cblas_ssymv(CblasColMajor, Uplo,
                *n, *alpha, A,
                *lda, X, *incx,
                *beta, Y, *incy);
}