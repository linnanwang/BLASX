/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "ZHER "

void cblas_zher(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const void *X, const int incX,
                void *A, const int ldA)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (ldA < MAX(1,N))                                    info = 7;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zher_p == NULL) blasx_init_cblas_func(&cblas_zher_p, "cblas_zher");
    Blasx_Debug_Output("Calling cblas_zher\n ");
    (*cblas_zher_p)(Order,Uplo,N,alpha,X,incX,A,ldA);
}

/* f77 interface */
void zher_(char *uplo, int *n, double *alpha, double *X, int *incx,
           double *A, int *lda)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling zher_ interface\n");
    cblas_zher(CblasColMajor,Uplo,
               *n, *alpha, X, *incx,
               A, *lda);
}