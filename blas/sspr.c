/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "SSPR "

void cblas_sspr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *Ap)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sspr_p == NULL) blasx_init_cblas_func(&cblas_sspr_p, "cblas_sspr");
    Blasx_Debug_Output("Calling cblas_sspr\n ");
    (*cblas_sspr_p)(Order,Uplo,N,alpha,X,incX,Ap);
}

/* f77 interface */
void sspr_(char *uplo, int *n, float *alpha, float *X, int *incx,
           float *Ap)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling sspr_ interface\n");
    cblas_sspr(CblasColMajor, Uplo,
                *n, *alpha, X,
                *incx, Ap);
}