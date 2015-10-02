/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "SSPR2 "

void cblas_sspr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *X,
                 const int incX, const float *Y, const int incY, float *A)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (incY == 0)                                         info = 7;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sspr2_p == NULL) blasx_init_cblas_func(&cblas_sspr2_p, "cblas_sspr2");
    Blasx_Debug_Output("Calling cblas_sspr2\n ");
    (*cblas_sspr2_p)(Order,Uplo,N,alpha,X,incX,Y,incY,A);
}

/* f77 interface */
void sspr2_(char *uplo, int *n, float *alpha,
            float *X, int *incx, float *Y, int *incy, float *A)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling sspr2_ interface\n");
    cblas_sspr2(CblasColMajor, Uplo,
                *n, *alpha, X,
                *incx, Y, *incy, A);
}