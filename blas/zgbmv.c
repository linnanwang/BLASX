/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZGBMV "

void cblas_zgbmv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha,
                 const void *A, const int lda, const void *X,
                 const int incX, const void *beta, void *Y, const int incY)
{
    cublasOperation_t transa;
    cublasStatus_t status;
    /*---error handler---*/
    int info = 0;
    if (CBLasTransToCuBlasTrans(TransA,&transa) <  0)           info = 1;
    else if (M < 0)                                             info = 2;
    else if (N < 0)                                             info = 3;
    else if (KL < 0)                                            info = 4;
    else if (KU < 0)                                            info = 5;
    else if (lda < (KU+KL+1))                                   info = 8;
    else if (incX == 0)                                         info = 10;
    else if (incY == 0)                                         info = 13;

    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zgbmv_p == NULL) blasx_init_cblas_func(&cblas_zgbmv_p, "cblas_zgbmv");
    Blasx_Debug_Output("Calling cblas_zgbmv\n ");
    (*cblas_zgbmv_p)(Order,TransA,M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void zgbmv_(char *transa,int *m, int *n, int *kl, int *ku, blasxDoubleComplexF77 *alpha, blasxDoubleComplexF77 *A, int *lda,
                     blasxDoubleComplexF77 *X, int *incx, blasxDoubleComplexF77 *beta, blasxDoubleComplexF77 *Y, int *incy)
{
    Blasx_Debug_Output("Calling zgbmv interface\n");
    enum CBLAS_TRANSPOSE TransA;
    int info = -1;
    if(F77TransToCBLASTrans(transa,&TransA) < 0)    info = 2;
    if (info >= 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_zgbmv(CblasColMajor,
                TransA, *m, *n,
                *kl, *ku, alpha,
                A, *lda, X,
                *incx, beta, Y, *incy);
}
