/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "DSPR "

void cblas_dspr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *Ap)
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
    if (cblas_dspr_p == NULL) blasx_init_cblas_func(&cblas_dspr_p, "cblas_dspr");
    Blasx_Debug_Output("Calling cblas_dspr\n ");
    (*cblas_dspr_p)(Order,Uplo,N,alpha,X,incX,Ap);
}

/* f77 interface */
void dspr_(char *uplo, int *n, double *alpha, double *X, int *incx,
           double *Ap)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling dspr_ interface\n");
    cblas_dspr(CblasColMajor, Uplo,
                *n, *alpha, X,
                *incx, Ap);
}