/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#include <math.h>
#define ERROR_NAME "ZGEMV "

void cblas_zgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda,
                 const blasxDoubleComplex *X, const int incX, const blasxDoubleComplex *beta,
                 blasxDoubleComplex *Y, const int incY)
{
    cublasOperation_t transa;
    cublasStatus_t status;
    /*---error handler---*/
    int info = 0;
    if (CBLasTransToCuBlasTrans(TransA,&transa) <  0)           info = 1;
    else if (M < 0)                                             info = 2;
    else if (N < 0)                                             info = 3;
    else if (lda < fmax(1, M))                                  info = 6;
    else if (incX == 0)                                         info = 8;
    else if (incY == 0)                                         info = 11;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    /*-------------------*/
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zgemv_p == NULL) blasx_init_cblas_func(&cblas_zgemv_p, "cblas_zgemv");
    Blasx_Debug_Output("Calling cblas_zgemv\n ");
    (*cblas_zgemv_p)(Order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void zgemv_(char *transa, int *m, int *n, blasxDoubleComplexF77 *alpha, blasxDoubleComplexF77 *A, int *lda,
                     blasxDoubleComplexF77 *x, int *incx, blasxDoubleComplexF77 *beta, blasxDoubleComplexF77 *y, int *incy)
{
    Blasx_Debug_Output("Calling zgemv interface\n");
    enum CBLAS_TRANSPOSE TransA;
    int info = -1;
    if(F77TransToCBLASTrans(transa,&TransA) < 0)    info = 2;
    if (info >= 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_zgemv(CblasColMajor,
                TransA, *m, *n,
                alpha, A, *lda,
                x, *incx, beta,
                y, *incy);
}
