/*
 * -- BLASX (version 1.0.0) --
 * @precisions normal z -> s d c
 */

#include "blasx.h"
#define ERROR_NAME "ZTRSV "

void cblas_ztrsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, void *X,
                 const int incX)
{
    cublasOperation_t transa; cublasFillMode_t uplo;
    cublasDiagType_t diag;
    /*---error handler---*/
    int info = 0;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)          info = 1;
    else if (CBLasTransToCuBlasTrans(TransA,&transa) < 0)           info = 2;
    else if (CBlasDiagModeToCuBlasDiagMode(Diag, &diag) < 0)        info = 3;
    else if (N < 0)                                                 info = 4;
    else if (lda < MAX(1,N))                                        info = 6;
    else if (incX == 0)                                             info = 8;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    /*-------------------*/
    //using cpu blas
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_ztrsv_p == NULL) blasx_init_cblas_func(&cblas_ztrsv_p, "cblas_ztrsv");
    Blasx_Debug_Output("Calling cblas_ztrsv\n ");
    (*cblas_ztrsv_p)(Order,Uplo,TransA,Diag,N,A,lda,X,incX);
}

/* f77 interface */
void ztrsv_(char *uplo, char *transa, char *diag, int *n, double *A, int *lda,
                      double *X, int *incx)
{
    Blasx_Debug_Output("Calling ztrsv_ interface\n");
    enum CBLAS_TRANSPOSE TransA; enum CBLAS_UPLO Uplo;
    enum CBLAS_DIAG Diag;
    int info = 0;
    if (F77DiagToCBLASDiag(diag, &Diag) < 0)                info =  3;
    if (F77TransToCBLASTrans(transa,&TransA) < 0)           info =  2;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_ztrsv(CblasColMajor, Uplo,
                TransA, Diag,
                *n, A, *lda, X,
                *incx);
}
