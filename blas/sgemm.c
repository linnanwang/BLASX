/*
 * @generated s Sat Jun 13 11:54:33 2015
 */
//special condition, n,m,k = 0 or alpha = 0, use cpu blas
#include <LRU.h>
#include <cblas.h>
#include <flops.h>
#include <blasx.h>
#include <blasx_sgemm.h>
#include <blasx_config.h>
#include <blasx_internal.h>
#include <blasx_tile_resource.h>
#include <blasx_globalpointers.h> //FOR CPU BLAS
#define ERROR_NAME "SGEMM "

void cblas_sgemm(const enum CBLAS_ORDER Order, 
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, 
                 const int M, const int N, const int K, 
                 const float alpha, const float *A, const int lda, 
                 const float *B, const int ldb,
                 const float beta, float *C, const int ldc )
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
    int nrowc = M;
    int ncolc = N;
    int info = 0;
    if (CBLasTransToCuBlasTrans(TransA,&transa) <  0) info = 1;
    else if (CBLasTransToCuBlasTrans(TransB,&transb) < 0) info = 2;
    else if (M < 0) info = 3;
    else if (N < 0) info = 4;
    else if (K < 0) info = 5;
    else if (lda < MAX(1, nrowa)) info = 8;
    else if (ldb < MAX(1, nrowb)) info = 10;
    else if (ldc < MAX(1, M)) info = 13;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    /*-------------------*/
    
    /*----dispatcher-----*/
    int type = 0; //1:cpu 2:cublasxt 3:blasx
    if (M <= 0 || N <= 0 || K <= 0) type = 1;
    if (type == 0 && (M > 1000 || N > 1000 || K > 1000)) type = 3;
    else                                                 type = 1;
    //Blasx_Debug_Output("type after dispatcher:%d\n",type);
    /*-------------------*/
    

    switch (type) {
        case 1:
        CPU_BLAS:
            Blasx_Debug_Output("calling cblas_sgemm:");
            if (cpublas_handle == NULL) blasx_init(CPU);
            if (cblas_sgemm_p == NULL) blasx_init_cblas_func(&cblas_sgemm_p, "cblas_sgemm");
            (*cblas_sgemm_p)(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
            break;
        case 2:
            if (cublasXt_handle == NULL) blasx_init(CUBLASXT);
            Blasx_Debug_Output("calling cublasSgemmXt:");
            status = cublasXtSgemm(cublasXt_handle,
                                   transa, transb,
                                   M, N, K,
                                   (float*)&alpha, (float*)A, lda,
                                   (float*)B, ldb,
                                   (float*)&beta, (float*)C, ldc);
            if( status != CUBLAS_STATUS_SUCCESS ) goto CPU_BLAS;
            break;
        case 3:
            Blasx_Debug_Output("calling BLASX:\n");
            cudaHostRegister(A,sizeof(float)*nrowa*ncola,cudaHostRegisterPortable);
            cudaHostRegister(B,sizeof(float)*nrowb*ncolb,cudaHostRegisterPortable);
            cudaHostRegister(C,sizeof(float)*nrowc*ncolc,cudaHostRegisterPortable);
#ifdef BENCHMARK
            double Gflops = FLOPS_DGEMM(M, N, K)/(1000000000);
            double gpu_start, gpu_end;
            gpu_start = get_cur_time();
#endif
            if (is_blasx_enable == 0) blasx_init(BLASX);
            assert( is_blasx_enable == 1 );
            assert( SYS_GPUS > 0 );
            assert( event_SGEMM[0] != NULL );
            assert( C_dev_SGEMM[0] != NULL );
            assert( handles_SGEMM[0] != NULL );
            assert( streams_SGEMM[0] != NULL );
            LRU_t* LRUs[10];
            int GPU_id = 0;
            for (GPU_id = 0; GPU_id < SYS_GPUS; GPU_id++)    LRUs[GPU_id] = LRU_init( GPU_id );
            blasx_sgemm(SYS_GPUS, handles_SGEMM, LRUs,
                        TransA, TransB,
                        M, N, K, alpha,
                        A, lda,
                        B, ldb,
                        beta,
                        C, ldc);
            for (GPU_id = 0; GPU_id < SYS_GPUS; GPU_id++)    LRU_free( LRUs[GPU_id], GPU_id );
#ifdef BENCHMARK
            gpu_end = get_cur_time();
            printf("BLASX (M:%5d,N:%5d,K:%5d) Speed:%9.1f type:%2d\n", M, N, K, (double)Gflops/(gpu_end - gpu_start), type);
#endif
            cudaHostUnregister(A);
            cudaHostUnregister(B);
            cudaHostUnregister(A);
            break;
        default:
            break;
    }

    //Blasx_Debug_Output("eventually use type:%d to compute\n",type);
}

/* f77 interface */
void sgemm_(char *transa, char *transb, 
            int *m, int *n, int *k, 
            float *alpha, float *A, int *lda,
            float *B, int *ldb,
            float *beta,  float *C, int *ldc)
{
    Blasx_Debug_Output("Calling sgemm interface\n");
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
    cblas_sgemm(CblasColMajor,
                TransA, TransB, 
                *m, *n, *k, 
                (float)*alpha, (float *)A, *lda, 
                (float *)B, *ldb, 
                (float)*beta,  (float *)C, *ldc);
}
