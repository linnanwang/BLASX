/*
 * -- BLASX (version 1.0.0) --
 * @precisions normal z -> s d c
 */

#include "blasx.h"
#define ERROR_NAME "DTBSV "

void cblas_stbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda,
                 float *X, const int incX)
{
    cublasOperation_t transa; cublasFillMode_t uplo;
    cublasDiagType_t diag;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)          info = 1;
    else if (CBLasTransToCuBlasTrans(TransA,&transa) < 0)           info = 2;
    else if (CBlasDiagModeToCuBlasDiagMode(Diag, &diag) < 0)        info = 3;
    else if (N < 0)                                                 info = 4;
    else if (K < 0)                                                 info = 5;
    else if (lda < (K+1))                                           info = 7;
    else if (incX == 0)                                             info = 9;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    /*-------------------*/
    //using cpu blas
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_stbsv_p == NULL) blasx_init_cblas_func(&cblas_stbsv_p, "cblas_stbsv");
    Blasx_Debug_Output("Calling cblas_stbsv\n ");
    (*cblas_stbsv_p)(Order,Uplo,TransA,Diag,N,K,A,lda,X,incX);
}

/* f77 interface */
void stbsv_(char *uplo, char *transa, char *diag, int *n, int *k, float *A, int *lda, float *X, int *incx)
{
    Blasx_Debug_Output("Calling ztbsv_ interface\n");
    enum CBLAS_TRANSPOSE TransA; enum CBLAS_UPLO Uplo;
    enum CBLAS_DIAG Diag; enum CBLAS_SIDE Side;
    int info = 0;
    if (F77DiagToCBLASDiag(diag, &Diag) < 0)                info =  3;
    if (F77TransToCBLASTrans(transa,&TransA) < 0)           info =  2;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_stbsv(CblasColMajor, Uplo,
                TransA, Diag,
                *n, *k, A, *lda,
                X,*incx);
}
