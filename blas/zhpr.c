/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHPR "

void cblas_zhpr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const void *X,
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
    if (cblas_zhpr_p == NULL) blasx_init_cblas_func(&cblas_zhpr_p, "cblas_zhpr");
    Blasx_Debug_Output("Calling cblas_zhpr\n ");
    (*cblas_zhpr_p)(Order,Uplo,N,alpha,X,incX,A);
}

/* f77 interface */
void zhpr_ (char *uplo, int *n, double *alpha, double *X, int *incx, double *A)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zhpr_ interface\n");
    cblas_zhpr(CblasColMajor, Uplo,
                *n, *alpha, X,
                *incx,A);
}