/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "DSPMV "

//SBMV
void cblas_dspmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *Ap,
                 const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 6;
    else if (incY == 0)                                         info = 9;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dspmv_p == NULL) blasx_init_cblas_func(&cblas_dspmv_p, "cblas_dspmv");
    Blasx_Debug_Output("Calling cblas_dspmv\n ");
    (*cblas_dspmv_p)(Order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);
}

/* f77 interface */
void dspmv_(char *uplo, int *n, double *alpha, double *Ap,
            double *X, int *incx, double *beta, double *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling dspmv_ interface\n");
    cblas_dspmv(CblasColMajor, Uplo,
                *n, *alpha, Ap,
                X, *incx,
                *beta, Y, *incy);
}