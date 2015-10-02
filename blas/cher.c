/*
 * @precisions normal z -> c d s
 */

#include "blasx.h"
#define ERROR_NAME "CHER "

void cblas_cher(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X, const int incX,
                void *A, const int lda)
{
    cublasFillMode_t uplo;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)      info = 1;
    else if (N < 0)                                             info = 2;
    else if (incX == 0)                                         info = 5;
    else if (lda < MAX(1,N))                                    info = 7;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_cher_p == NULL) blasx_init_cblas_func(&cblas_cher_p, "cblas_cher");
    Blasx_Debug_Output("Calling cblas_cher\n ");
    (*cblas_cher_p)(Order,Uplo,N,alpha,X,incX,A,lda);
}

/* f77 interface */
void cher_(char *uplo, int *n, float *alpha, float *X, int *incx,
           float *A, int *lda)
{
    enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                     info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("Calling cher_ interface\n");
    cblas_cher(CblasColMajor,Uplo,
               *n, *alpha, X, *incx,
               A, *lda);
}