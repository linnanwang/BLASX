/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "DSYR2 "

void cblas_dsyr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *X,
                 const int incX, const double *Y, const int incY, double *A,
                 const int lda)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (incY == 0)                                         info = 7;
    else if (lda < MAX(1,N))                                    info = 9;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dsyr2_p == NULL) blasx_init_cblas_func(&cblas_dsyr2_p, "cblas_dsyr2");
    Blasx_Debug_Output("Calling cblas_dsyr2\n ");
    (*cblas_dsyr2_p)(Order,Uplo,N,alpha,X,incX,Y,incY,A,lda);
}

/* f77 interface */
void dsyr2_(char *uplo, int *n, double *alpha,
            double *X, int *incx, double *Y, int *incy, double *A, int *lda)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling dsymv_ interface\n");
    cblas_dsyr2(CblasColMajor, Uplo,
                *n, *alpha, X,
                *incx, Y, *incy, A,
                *lda);
}