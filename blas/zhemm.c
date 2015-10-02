/*
   -- BLASX (version 1.0.0) --
 * @precisions normal z -> c d s
 */

#include <LRU.h>
#include <cblas.h>
#include <flops.h>
#include <blasx.h>
#include <blasx_config.h>
#include <blasx_internal.h>
#include <blasx_tile_resource.h>
#include <blasx_globalpointers.h> //FOR CPU BLAS
#define ERROR_NAME "ZHEMM"

void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                      const enum CBLAS_UPLO Uplo,   const int M,
                      const int N,      const void *alpha,
                      const void *A,    const int lda,
                      const void *B,    const int ldb,
                      const void *beta, void *C,
                      const int ldc )
{
    cublasOperation_t transa; cublasFillMode_t uplo;
    cublasSideMode_t side; cublasDiagType_t diag;
    cublasStatus_t status;
    
    /*---error handler---*/
    int nrowa;
    int info = 0;
    if (Side == CblasLeft) nrowa = M;
    else nrowa = N;
    if(ldc < MAX(1,M))                                        info = 12;
    if(ldb < MAX(1,M))                                        info = 9;
    if(lda < MAX(1,nrowa))                                    info = 7;
    if (N < 0)                                                info = 4;
    if (M < 0)                                                info = 3;
    if (CBlasFilledModeToCuBlasFilledMode(Uplo,&uplo) < 0)    info = 2;
    if (CBlasSideToCuBlasSide(Side,&side) < 0)                info = 1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    /*-------------------*/
    
    /*----dispatcher-----*/
    int type = 0; //1:cpu 2:cublasxt 3:blasx
    if (N <= 0 || M <= 0)                      type = 1;
    if (type == 0 && (N > 1000 || M > 1000))   type = 1;
    else                                       type = 1;
    /*-------------------*/

    switch (type) {
        case 1:
        CPU_BLAS:
            if (cpublas_handle == NULL) blasx_init(CPU);
            if (cblas_zhemm_p == NULL) blasx_init_cblas_func(&cblas_zhemm_p, "cblas_zhemm");
            Blasx_Debug_Output("Calling cblas_zhemm\n ");
            (*cblas_zhemm_p)(Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("Calling cublasXtZhemm\n");
            status = cublasXtZhemm(cublasXt_handle,
                                   side,   uplo,
                                   M, N,
                                   (cuDoubleComplex*)alpha,
                                   (cuDoubleComplex*)A, lda,
                                   (cuDoubleComplex*)B, ldb,
                                   (cuDoubleComplex*)beta,
                                   (cuDoubleComplex*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
        default:
            break;
    }
}

/* f77 interface */
void zhemm_(char *side, char *uplo,
            int *m, int *n,
            blasxDoubleComplexF77 *alpha, blasxDoubleComplexF77 *A, int *lda,
            blasxDoubleComplexF77 *B, int *ldb,
            blasxDoubleComplexF77 *beta,  blasxDoubleComplexF77 *C, int *ldc)
{
    Blasx_Debug_Output("Calling zhemm_ interface\n");
    enum CBLAS_UPLO Uplo;   enum CBLAS_SIDE Side;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  2;
    if (F77SideToCBlasSide(side, &Side) < 0)                info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_zhemm(CblasColMajor,Side,Uplo,
                *m,*n,alpha,
                A,*lda,B,*ldb,
                beta,C,*ldc);
}
