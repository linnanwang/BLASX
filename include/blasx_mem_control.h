#ifndef STANDARD_LIB_H
#define STANDARD_LIB_H
#include <stdio.h>
#include <stdlib.h>
#endif

#ifndef MEM_CONTROL_KERNEL_H
#define MEM_CONTROL_KERNEL_H
#include <LRU.h>
#include <cblas.h>

#define ABS(x)  (((x) < 0) ? -(x) : (x))
typedef struct reader_tracker {
    void *addr;
    int GPU_id;
    int is_trans_done;
}reader_tracker;

void atomic_reader_minus(volatile rbt_node* node);
void atomic_reader_plus(volatile rbt_node* node);
void margin_adjustment(int row_original_matrix, int col_original_matrix, int block_dim, int block_row, int block_col, int *nrow_dev, int *ncol_dev);
void blasx_get_index(int tile_index, int row_start, int row_end, int* x, int* y, enum CBLAS_UPLO uplo, int ROWs);
void mem_control_kernel_double(double *starting_point_A, double **A_dev,
                        LRU_t **LRUs, const int GPUs, const int GPU_id, int block_dim,
                        int *mem_cpy_counter, reader_tracker *addr_track,
                        cudaStream_t *stream,
                        int nrowa_dev, int ncola_dev, int lda);
void mem_control_kernel_float(float *starting_point_A, float **A_dev,
                              LRU_t **LRUs, const int GPUs, const int GPU_id, int block_dim,
                              int *mem_cpy_counter, reader_tracker *addr_track,
                              cudaStream_t *stream,
                              int nrowa_dev, int ncola_dev, int lda);
/*--GEMM--*/
void collect_final_result_dgemm(int *tasks_rs, int *tasks_rs_size, int switcher, cudaStream_t *stream, double** C_dev, int block_dim, int stream_num, int x, int y, int z, int nrowc, int ncolc, int ldc, double *C);
void collect_final_result_sgemm(int *tasks_rs, int *tasks_rs_size, int switcher, cudaStream_t *stream, float** C_dev, int block_dim, int stream_num, int x, int y, int z, int nrowc, int ncolc, int ldc, float *C);
/*--SYRK&SYR2K--*/
void collect_final_result_dsyrk_syr2k(int *tasks_rs, int *tasks_rs_size, int switcher, cudaStream_t *stream, double** C_dev, int block_dim, int stream_num, int x, int y, int z, int nrowc, int ncolc, int ldc, double *C, enum CBLAS_UPLO Uplo);

#endif