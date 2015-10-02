/*
 * @generated c Mon Jun 15 16:57:40 2015
 */

#include <LRU.h>
#include <cblas.h>
#include <flops.h>
#include <blasx.h>
#include <blasx_config.h>
#include <blasx_internal.h>
#include <blasx_tile_resource.h>
#include <blasx_globalpointers.h> //FOR CPU BLAS
#define ERROR_NAME "CSYRK "

void cblas_csyrk(const enum CBLAS_ORDER Order, 
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, 
                 const int N, const int K,
                 const blasxFloatComplex *alpha, const blasxFloatComplex *A, const int lda,
                 const blasxFloatComplex *beta, blasxFloatComplex *C, const int ldc)
{
    cublasOperation_t trans; cublasFillMode_t uplo;
    cublasStatus_t status;
    /*---error handler---*/
    int info = 0;
    int nrowa, ncola;
    if (Trans == CblasNoTrans){
        nrowa = N;
        ncola = K;
    }else{
        nrowa = K;
        ncola = N;
    }
    if (ldc < MAX(1,N))                                     info = 10;
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
            Blasx_Debug_Output("Calling cblas_csyrk\n ");
            if (cpublas_handle == NULL) blasx_init(CPU);
            if (cblas_csyrk_p == NULL)  blasx_init_cblas_func(&cblas_csyrk_p, "cblas_csyrk");
            (*cblas_csyrk_p)(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("Calling cublasXtCsyrk\n");
            status = cublasXtCsyrk(cublasXt_handle,
                                   uplo, trans,
                                   N, K,
                                   (cuFloatComplex*)alpha, (cuFloatComplex*)A, lda,
                                   (cuFloatComplex*)beta, (cuFloatComplex*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
        default:
            break;
    }
}

/* f77 interface */
void csyrk_(char *uplo, char *trans,
            int *n, int *k,
            blasxFloatComplexF77 *alpha, blasxFloatComplexF77 *A, int *lda,
            blasxFloatComplexF77 *beta,  blasxFloatComplexF77 *C, int *ldc)
{
    Blasx_Debug_Output("Calling csyrk_ interface\n");
    enum CBLAS_TRANSPOSE Trans; enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)         info =  1;
    if (F77TransToCBLASTrans(trans,&Trans) < 0)     info =  2;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    Blasx_Debug_Output("uplo:%c trans:%c n:%d k:%d alpha:%f beta:%f lda:%d ldc:%d\n",*uplo,*trans,*n,*k,*alpha,*beta,*lda,*ldc);
    cblas_csyrk(CblasColMajor, Uplo,
                Trans, *n, *k,
                (blasxFloatComplex*)alpha, (blasxFloatComplex*)A, *lda,
                (blasxFloatComplex*)beta, (blasxFloatComplex*)C, *ldc);
}
