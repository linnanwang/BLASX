/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "CHEMV "

void cblas_chemv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
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
    if (cblas_chemv_p == NULL) blasx_init_cblas_func(&cblas_chemv_p, "cblas_chemv");
    Blasx_Debug_Output("Calling cblas_chemv\n ");
    (*cblas_chemv_p)(Order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void chemv_(char *uplo, int *n, float *alpha, float *A, int *lda,
            float *X, int *incx, float *beta, float *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zhemv_ interface\n");
    cblas_chemv(CblasColMajor,Uplo,
                *n, alpha, A,
                *lda, X, *incx,
                beta,Y, *incy);
}
