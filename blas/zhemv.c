/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHEMV "

void cblas_zhemv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *A,
                 const int ldA, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (ldA < MAX(1,N))                                    info = 5;
    else if (incX == 0)                                         info = 7;
    else if (incY == 0)                                         info = 10;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zhemv_p == NULL) blasx_init_cblas_func(&cblas_zhemv_p, "cblas_zhemv");
    Blasx_Debug_Output("Calling cblas_zhemv\n ");
    (*cblas_zhemv_p)(Order,Uplo,N,alpha,A,ldA,X,incX,beta,Y,incY);
}

/* f77 interface */
void zhemv_(char *uplo, int *n, double  *alpha, double *A, int *lda,
            double *X, int *incx, double *beta, double *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zhemv_ interface\n");
    cblas_zhemv(CblasColMajor,Uplo,
                *n, alpha, A,
                *lda, X, *incx,
                beta,Y, *incy);
}
