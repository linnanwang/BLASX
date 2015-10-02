/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZGERU "

void cblas_zgeru(const enum CBLAS_ORDER Order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
    cublasOperation_t transa;
    cublasStatus_t status;
    /*---error handler---*/
    int info = 0;
    if (M < 0)                                                  info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (incY == 0)                                         info = 7;
    else if (lda <= MAX(1,M))                                   info = 9;

    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zgeru_p == NULL) blasx_init_cblas_func(&cblas_zgeru_p, "cblas_zgeru");
    Blasx_Debug_Output("Calling cblas_zgeru\n ");
    (*cblas_zgeru_p)(Order,M,N,alpha,X,incX,Y,incY,A,lda);
}

/* f77 interface */
void zgeru_(int *m, int *n, double *alpha, double *X, int *incx,
            double *Y, int *incy, double *A, int *lda)
{
    Blasx_Debug_Output("Calling zgeru interface\n");
    cblas_zgeru(CblasColMajor, *m, *n,
                alpha, X, *incx,
                Y, *incy, A, *lda);
}
