/*
 * @generated s Mon Jun 15 16:57:41 2015
 */

#include <LRU.h>
#include <cblas.h>
#include <flops.h>
#include <blasx.h>
#include <blasx_config.h>
#include <blasx_internal.h>
#include <blasx_tile_resource.h>
#include <blasx_globalpointers.h> //FOR CPU BLAS
#define ERROR_NAME "SSYMM "

//SSYMM
void cblas_ssymm(const enum CBLAS_ORDER Order, 
                 const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, 
                 const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta,
                 float *C, const int ldc)
{
    cublasSideMode_t side; cublasFillMode_t uplo;
    cublasStatus_t status;
    /*---error handler---*/
    int nrowa;
    if (Side == CblasLeft) nrowa = M;
    else nrowa = N;
    int info = 0;
    if (CBlasSideToCuBlasSide(Side, &side) < 0)                     info = 1;
    else if (CBlasFilledModeToCuBlasFilledMode(Uplo, &uplo) < 0)    info = 2;
    else if (M < 0)                                                 info = 3;
    else if (N < 0)                                                 info = 4;
    else if (lda < MAX(1, nrowa))                                   info = 7;
    else if (ldb < MAX(1, M))                                       info = 9;
    else if (ldc < MAX(1, M))                                       info = 12;
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
            if (cblas_ssymm_p == NULL) blasx_init_cblas_func(&cblas_ssymm_p, "cblas_ssymm");
            Blasx_Debug_Output("Calling cblas_ssymm\n");
            (*cblas_ssymm_p)(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("Calling cublasXtSsymm\n");
            status = cublasXtSsymm(cublasXt_handle,
                                   side, uplo,
                                   M, N,
                                   (float*)&alpha, (float*)A, lda,
                                   (float*)B, ldb,
                                   (float*)&beta, (float*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
        default:
            break;
    }
}
//f77 interface
void ssymm_(char *side, char *uplo, 
            int *m, int *n, 
            float *alpha, float *A, int *lda, 
            float *B, int *ldb, 
            float *beta, float *C, int *ldc)
{
    Blasx_Debug_Output("Called symm interface\n");
    enum CBLAS_SIDE Side; enum CBLAS_UPLO Uplo;
    int info = 0;
    if (F77UploToCBlasUplo(uplo,&Uplo) < 0)     info =  2;
    if (F77SideToCBlasSide(side, &Side) < 0)    info =  1;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_ssymm(CblasColMajor, 
                Side, Uplo, 
                *m, *n, 
                (float)*alpha, (float *)A, *lda, 
                (float *)B, *ldb, 
                (float)*beta, (float *)C, *ldc);
}

