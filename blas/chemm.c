/*
   -- BLASX (version 1.0.0) --
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
#define ERROR_NAME "CHEMM"

void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
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
            if (cblas_chemm_p == NULL) blasx_init_cblas_func(&cblas_chemm_p, "cblas_chemm");
            Blasx_Debug_Output("Calling cblas_chemm\n ");
            (*cblas_chemm_p)(Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("Calling cublasXtChemm\n");
            status = cublasXtChemm(cublasXt_handle,
                                   side,   uplo,
                                   M, N,
                                   (cuFloatComplex*)alpha,
                                   (cuFloatComplex*)A, lda,
                                   (cuFloatComplex*)B, ldb,
                                   (cuFloatComplex*)beta,
                                   (cuFloatComplex*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
        default:
            break;
    }
}

/* f77 interface */
void chemm_(char *side, char *uplo,
            int *m, int *n,
            blasxFloatComplexF77 *alpha, blasxFloatComplexF77 *A, int *lda,
            blasxFloatComplexF77 *B, int *ldb,
            blasxFloatComplexF77 *beta,  blasxFloatComplexF77 *C, int *ldc)
{
    Blasx_Debug_Output("Calling chemm_ interface\n");
    enum CBLAS_UPLO Uplo;   enum CBLAS_SIDE Side;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)                 info =  2;
    if (F77SideToCBlasSide(side, &Side) < 0)                info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_chemm(CblasColMajor,Side,Uplo,
                *m,*n,alpha,
                A,*lda,B,*ldb,
                beta,C,*ldc);
}
