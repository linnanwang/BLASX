/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "DSYMV "

void cblas_dsymv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY)
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
    if (cblas_dsymv_p == NULL) blasx_init_cblas_func(&cblas_dsymv_p, "cblas_dsymv");
    Blasx_Debug_Output("Calling cblas_dsymv\n ");
    (*cblas_dsymv_p)(Order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/* f77 interface */
void dsymv_(char *uplo, int *n, double *alpha, double *A, int *lda,
            double *X, int *incx, double *beta, double *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling dsymv_ interface\n");
    cblas_dsymv(CblasColMajor, Uplo,
                *n, *alpha, A,
                *lda, X, *incx,
                *beta, Y, *incy);
}