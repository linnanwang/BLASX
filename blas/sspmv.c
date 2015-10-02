/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "SSPMV "

//SBMV
void cblas_sspmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *Ap,
                 const float *X, const int incX,
                 const float beta, float *Y, const int incY)
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
    if (cblas_sspmv_p == NULL) blasx_init_cblas_func(&cblas_sspmv_p, "cblas_sspmv");
    Blasx_Debug_Output("Calling cblas_sspmv\n ");
    (*cblas_sspmv_p)(Order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);
}

/* f77 interface */
void sspmv_(char *uplo, int *n, float *alpha, float *Ap,
            float *X, int *incx, float *beta, float *Y, int *incy)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling dspmv_ interface\n");
    cblas_sspmv(CblasColMajor, Uplo,
                *n, *alpha, Ap,
                X, *incx,
                *beta, Y, *incy);
}