/*
 * @generated c Thu Apr 10 19:58:24 2014
 */

#include "blasx.h"
#include <math.h>
#define ERROR_NAME "ZGEMV "

void cblas_cgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
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
    if (cblas_cgemv_p == NULL) blasx_init_cblas_func(&cblas_cgemv_p, "cblas_cgemv");
    Blasx_Debug_Output("Calling cblas_cgemv\n ");
    (*cblas_cgemv_p)(Order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void cgemv_(char *transa, int *m, int *n, float *alpha, float *A, int *lda,
            float *x, int *incx, float *beta, float *y, int *incy)
{
    Blasx_Debug_Output("Calling cgemv interface\n");
    enum CBLAS_TRANSPOSE TransA;
    int info = -1;
    if(F77TransToCBLASTrans(transa,&TransA) < 0)    info = 2;
    if (info >= 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_cgemv(CblasColMajor,
                TransA, *m, *n,
                alpha, A, *lda,
                x, *incx, beta,
                y, *incy);
}
