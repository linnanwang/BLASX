/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHPR "

void cblas_chpr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X,
                const int incX, void *A)
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
    if (cblas_chpr_p == NULL) blasx_init_cblas_func(&cblas_chpr_p, "cblas_chpr");
    Blasx_Debug_Output("Calling cblas_chpr\n ");
    (*cblas_chpr_p)(Order,Uplo,N,alpha,X,incX,A);
}

/* f77 interface */
void chpr_ (char *uplo, int *n, float *alpha, float *X, int *incx, float *A)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling chpr_ interface\n");
    cblas_chpr(CblasColMajor, Uplo,
                *n, *alpha, X,
                *incx,A);
}