#ifndef BLASX_TILE_RESOURCE
#define BLASX_TILE_RESOURCE
#include <blasx.h>
#include <blasx_common.h>
#include <blasx_internal.h>

/*--------cpublas-------*/
extern void* cpublas_handle;
extern char* blas_path;
/*-------cublasXt-------*/
extern cublasXtHandle_t cublasXt_handle;
/*---------blasx--------*/
extern int SYS_GPUS;
extern int is_blasx_enable;
//SGEMM
extern cublasHandle_t handles_SGEMM[10];
extern cudaStream_t   streams_SGEMM[40];
extern cudaEvent_t    event_SGEMM[40];
extern float*         C_dev_SGEMM[80];
//DGEMM
extern cublasHandle_t handles_DGEMM[10];
extern cudaStream_t   streams_DGEMM[40];
extern cudaEvent_t    event_DGEMM[40];
extern double*        C_dev_DGEMM[80];

void blasx_init_cblas_func(void **cblas_func_p, char *fun_name);

#endif /* BLASX_GLOBALPOINTERS */