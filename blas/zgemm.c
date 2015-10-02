/*
 * @precisions normal z -> c d s
 */
//special condition, n,m,k = 0 or alpha = 0, use cpu blas
#include <cblas.h>
#include <flops.h>
#include <blasx.h>
#include <blasx_internal.h>
#include <blasx_tile_resource.h>
#include <blasx_globalpointers.h>
#define ERROR_NAME "ZGEMM "

void cblas_zgemm(const enum CBLAS_ORDER Order, 
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, 
                 const int M, const int N, const int K, 
                 const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda, 
                 const blasxDoubleComplex *B, const int ldb,
                 const blasxDoubleComplex *beta, blasxDoubleComplex *C, const int ldc )
{
    cublasOperation_t transa, transb;
    cublasStatus_t status;
    /*---error handler---*/
    int nrowa, ncola, nrowb, ncolb;
    if (TransA == CblasNoTrans) {
        nrowa = M;
        ncola = K;
    } else {
        nrowa = K;
        ncola = M;
    }
    if (TransB == CblasNoTrans) {
        nrowb = K;
        ncolb = N;
    } else {
        nrowb = N;
        ncolb = K;
    }
    int info = 0;
    if (CBLasTransToCuBlasTrans(TransA,&transa) <  0) info = 1;
    else if (CBLasTransToCuBlasTrans(TransB,&transb) < 0) info = 2;
    else if (M < 0) info = 3;
    else if (N < 0) info = 4;
    else if (K < 0) info = 5;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
    }
    /*-------------------*/
    
    /*----dispatcher-----*/
    int type = 0; //1:cpu 2:cublasxt 3:blasx
    if (M <= 0 || N <= 0 || K <= 0)                      type = 1;
    if (type == 0 && (M > 1000 || N > 1000 || K > 1000)) type = 2; //WAITING IMPLEMENT
    else                                                 type = 1;
    /*-------------------*/

#ifdef BENCHMARK
    double Gflops = FLOPS_ZGEMM(M, N, K)/(1000000000);
    double gpu_start, gpu_end;
    gpu_start = get_cur_time();
#endif
    switch (type) {
        case 1:
        CPU_BLAS:
            Blasx_Debug_Output("calling cblas_zgemm:");
            if (cpublas_handle == NULL) blasx_init(CPU);
            if (cblas_zgemm_p == NULL) blasx_init_cblas_func(&cblas_zgemm_p, "cblas_zgemm");
            (*cblas_zgemm_p)(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("calling cublasZgemmXt:");
            status = cublasXtZgemm(cublasXt_handle,
                                   transa, transb,
                                   M, N, K,
                                   (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda,
                                   (cuDoubleComplex*)B, ldb,
                                   (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
//        case 3:
//            Blasx_Debug_Output("calling BLASX:\n");
//            if (is_blasx_enable == 0) blasx_init(BLASX);
//            assert( is_blasx_enable == 1 );
//            assert( SYS_GPUS > 0 );
//            assert( event_ZGEMM[0] != NULL );
//            assert( C_dev_ZGEMM[0] != NULL );
//            assert( handles_ZGEMM[0] != NULL );
//            assert( streams_ZGEMM[0] != NULL );
//            LRU_t* LRUs[10];
//            int GPU_id = 0;
//            for (GPU_id = 0; GPU_id < SYS_GPUS; GPU_id++)    LRUs[GPU_id] = LRU_init( GPU_id );
//            blasx_zgemm(SYS_GPUS, handles_ZGEMM, LRUs,
//                        TransA, TransB,
//                        M, N, K, alpha,
//                        A, lda,
//                        B, ldb,
//                        beta,
//                        C, ldc);
//            for (GPU_id = 0; GPU_id < SYS_GPUS; GPU_id++)    LRU_free( LRUs[GPU_id], GPU_id );
//            break;
        default:
            break;
    }
#ifdef BENCHMARK
    gpu_end = get_cur_time();
    printf("BLASX (M:%5d,N:%5d,K:%5d) Speed:%9.1f type:%2d\n", M, N, K, (double)Gflops/(gpu_end - gpu_start), type);
#endif

}

/* f77 interface */
void zgemm_(char *transa, char *transb, 
            int *m, int *n, int *k, 
            blasxDoubleComplexF77 *alpha, blasxDoubleComplexF77 *A, int *lda,
            blasxDoubleComplexF77 *B, int *ldb,
            blasxDoubleComplexF77 *beta,  blasxDoubleComplexF77 *C, int *ldc)
{
    Blasx_Debug_Output("Calling zgemm interface\n");
    Blasx_Debug_Output("transa:%c transb:%c m:%d n:%d k:%d alpha:%f lda:%d ldb:%d beta:%f ldc:%d\n",
                       *transa,*transb,*m,*n,*k,*alpha,*lda,*ldb,*beta,*ldc);
    enum CBLAS_TRANSPOSE TransA, TransB;
    int info = -1;
    if(F77TransToCBLASTrans(transa,&TransA) < 0)    info = 1;
    if(F77TransToCBLASTrans(transb,&TransB) < 0)    info = 2;
    if (info >= 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    cblas_zgemm(CblasColMajor,
                TransA, TransB, 
                *m, *n, *k, 
                (blasxDoubleComplex *)alpha, (blasxDoubleComplex *)A, *lda, 
                (blasxDoubleComplex *)B, *ldb, 
                (blasxDoubleComplex *)beta,  (blasxDoubleComplex *)C, *ldc);
}
