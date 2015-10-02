/*
 * @generated d Tue Apr  8 15:19:39 2014
 */

#include "blasx.h"
#include <math.h>
#define ERROR_NAME "DGEMV "

void cblas_dgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
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
    if (cblas_dgemv_p == NULL) blasx_init_cblas_func(&cblas_dgemv_p, "cblas_dgemv");
    Blasx_Debug_Output("Calling cblas_dgemv\n ");
    (*cblas_dgemv_p)(Order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void dgemv_(char *transa, int *m, int *n, double *alpha, double *A, int *lda,
                     double *x, int *incx, double *beta, double *y, int *incy)
{
    Blasx_Debug_Output("Calling dgemv interface\n");
    enum CBLAS_TRANSPOSE TransA;
    int info = -1;
    if(F77TransToCBLASTrans(transa,&TransA) < 0)    info = 2;
    if (info >= 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_dgemv(CblasColMajor,
                TransA, *m, *n,
                *alpha, A, *lda,
                x, *incx, *beta,
                y, *incy);
}
