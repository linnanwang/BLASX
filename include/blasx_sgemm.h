#ifndef BLASX_SGEMM
#define BLASX_SGEMM
#include <blasx.h>
#include <flops.h>
#include <pthread.h>
#include <blasx_affinity.h>
#include <blasx_task_queue.h>
#include <blasx_mem_control.h>
#include <blasx_tile_resource.h>


/*----matrix config----*/
typedef struct matrix_config {
    int x;
    int y;
    int z;
    int lda;
    int ldb;
    int ldc;
    void *A;
    void *B;
    void *C;
    int nrowa;
    int nrowb;
    int nrowc;
    int ncola;
    int ncolb;
    int ncolc;
    float alpha;
    float beta;
    int block_dim;
    enum CBLAS_TRANSPOSE TransA;
    enum CBLAS_TRANSPOSE TransB;
}matrix_config;
/*---CPU variables---*/
struct cpu_thread_data{
    matrix_config* mat_conf;
    queue_t* tasks_queue;
};
/*---GPU variables---*/
struct gpu_thread_data{
    int GPUs;
    int GPU_id;
    LRU_t **LRUs;
    queue_t* tasks_queue;
    cublasHandle_t* handles;
    matrix_config* mat_conf;
#ifdef thread_barrier
    pthread_barrier_t* barr;
#endif
};

int blasx_sgemm(const int GPUs, cublasHandle_t* handles, LRU_t **LRUs,
                enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                const int M, const int N, const int K, const float alpha,
                float *A, int lda,
                float *B, int ldb,
                const float beta,
                float *C, int ldc);

#endif /* BLASX_SGEMM */