/*
 * @generated s Thu Apr 10 14:35:11 2014
 */

#include "blasx.h"
#include <math.h>
#define ERROR_NAME "SGEMV "

void cblas_sgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY)
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
    if (cblas_sgemv_p == NULL) blasx_init_cblas_func(&cblas_sgemv_p, "cblas_sgemv");
    Blasx_Debug_Output("Calling cblas_sgemv\n ");
    (*cblas_sgemv_p)(Order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void sgemv_(char *transa, int *m, int *n, float *alpha, float *A, int *lda,
            float *x, int *incx, float *beta, float *y, int *incy)
{
    Blasx_Debug_Output("Calling sgemv interface\n");
    enum CBLAS_TRANSPOSE TransA;
    int info = -1;
    if(F77TransToCBLASTrans(transa,&TransA) < 0)    info = 2;
    if (info >= 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_sgemv(CblasColMajor,
                TransA, *m, *n,
                *alpha, A, *lda,
                x, *incx, *beta,
                y, *incy);
}
