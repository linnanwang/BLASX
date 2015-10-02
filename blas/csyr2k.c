/*
 * -- BLASX (version 1.0.0) --
 * @generated c Mon Jun 15 16:57:41 2015
 */

#include <LRU.h>
#include <cblas.h>
#include <flops.h>
#include <blasx.h>
#include <blasx_config.h>
#include <blasx_internal.h>
#include <blasx_tile_resource.h>
#include <blasx_globalpointers.h> //FOR CPU BLAS
#define ERROR_NAME "CSYR2K"

void cblas_csyr2k(const enum CBLAS_ORDER Order,
                 const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans,
                 const int N,
                 const int K,
                 const blasxFloatComplex *alpha,
                 const blasxFloatComplex *A,
                 const int lda,
                 const blasxFloatComplex *B,
                 const int ldb,
                 const blasxFloatComplex *beta,
                 blasxFloatComplex *C,
                 const int ldc)
{
    cublasOperation_t trans; cublasFillMode_t uplo;
    cublasStatus_t status;
    
    /*---error handler---*/
    int info = 0;
    int nrowa, ncola;
    if (Trans == CblasNoTrans){
        ncola = K;
        nrowa = N;
    }else{
        nrowa = K;
        ncola = N;
    }
    if (ldc < MAX(1,N))                                     info = 12;
    if (ldb < MAX(1,nrowa))                                 info =  9;
    if (lda < MAX(1,nrowa))                                 info =  7;
    if (K < 0)                                              info =  4;
    if (N < 0)                                              info =  3;
    if (CBLasTransToCuBlasTrans(Trans, &trans) < 0)         info =  2;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo, &uplo) < 0) info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    /*-------------------*/
    
    /*----dispatcher-----*/
    int type = 0; //1:cpu 2:cublasxt 3:blasx
    if (N <= 0 || K <= 0)                      type = 1;
    if (type == 0 && (N > 1000 || K > 1000))   type = 1;
    else                                       type = 1;
    /*-------------------*/
    
    switch (type) {
        case 1:
        CPU_BLAS:
            if (cpublas_handle == NULL) blasx_init(CPU);
            if (cblas_csyr2k_p == NULL) blasx_init_cblas_func(&cblas_csyr2k_p, "cblas_csyr2k");
            Blasx_Debug_Output("Calling cblas_csyr2k\n ");
            (*cblas_csyr2k_p)(Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("Calling cublasXtCsyr2k\n");
            status = cublasXtCsyr2k(cublasXt_handle,
                                    uplo, trans,
                                    N, K,
                                    (cuFloatComplex*)alpha, (cuFloatComplex*)A, lda,
                                    (cuFloatComplex*)B, ldb,
                                    (cuFloatComplex*)beta, (cuFloatComplex*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
        default:
            break;
    }
}

/* f77 interface */
void csyr2k_(char *uplo, char *trans,
            int *n, int *k,
            blasxFloatComplexF77 *alpha, blasxFloatComplexF77 *A, int *lda,
            blasxFloatComplexF77 *B, int *ldb,
            blasxFloatComplexF77 *beta,  blasxFloatComplexF77 *C, int *ldc)
{
    Blasx_Debug_Output("Calling csyr2k_ interface\n");
    enum CBLAS_TRANSPOSE Trans; enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)         info =  1;
    if (F77TransToCBLASTrans(trans,&Trans) < 0)     info =  2;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("uplo:%c trans:%c n:%d k:%d alpha:%f beta:%f lda:%d ldc:%d\n",*uplo,*trans,*n,*k,*alpha,*beta,*lda,*ldc);

    cblas_csyr2k(CblasColMajor,
                Uplo,
                Trans,
                *n,
                *k,
                (blasxFloatComplex*)alpha,
                (blasxFloatComplex*)A,
                *lda,
                (blasxFloatComplex*)B,
                *ldb,
                (blasxFloatComplex*)beta,
                (blasxFloatComplex*)C,
                *ldc);
}
