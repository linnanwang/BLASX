#include <blasx_tile_resource.h>

/*----cublasXt----*/
cublasXtHandle_t cublasXt_handle = NULL;
/*-----cpublas----*/
void* cpublas_handle = NULL;
char* blas_path      = CPUBLAS;
/*------mutex-----*/
/*---------blasx--------*/
int SYS_GPUS = 0;
int is_blasx_enable = 0;
//SGEMM
cublasHandle_t handles_SGEMM[10] = { NULL };
cudaStream_t   streams_SGEMM[40] = { NULL };
cudaEvent_t    event_SGEMM[40] = { NULL };
float*         C_dev_SGEMM[80] = { NULL };
//DGEMM
cublasHandle_t handles_DGEMM[10] = { NULL };
cudaStream_t   streams_DGEMM[40] = { NULL };
cudaEvent_t    event_DGEMM[40] = { NULL };
double*        C_dev_DGEMM[80] = { NULL };



void blasx_init_cblas_func(void **cblas_func_p, char *fun_name)
{
    blasx_mutex_lock();
    if (*cblas_func_p == NULL) {
        Blasx_Debug_Output("cblas function linked\n");
        *cblas_func_p = dlsym(cpublas_handle, fun_name);
        Blasx_Debug_Output("cblas %p\n", *cblas_func_p);
    }
    blasx_mutex_unlock();
}

void blasx_resource_dest(int GPUs, cublasHandle_t* handles, cudaStream_t* streams, cudaEvent_t* events, float** C_dev)
{
    int GPU_id;
    for (GPU_id = 0; GPU_id < GPUs; GPU_id++) {
        cudaSetDevice( GPU_id );
        int i = 0;
        for (i = 0; i < STREAMNUM; i++) {
            cudaStreamDestroy( streams[i+GPU_id*STREAMNUM] );
            cudaEventDestroy( events[i+GPU_id*STREAMNUM] );
        }
        for (i = 0; i < STREAMNUM*2; i++) cudaFree( C_dev[i+GPU_id*STREAMNUM*2] );
        cublasDestroy( handles[GPU_id] );
    }
}

/*------Destruction------*/
extern __attribute__ ((destructor)) void global_pointers_destruction(){
    printf("-----Destructing:\n");
    if(cublasXt_handle!=NULL)   {
        cublasStatus_t status=cublasXtDestroy(cublasXt_handle);
        if (status==CUBLAS_STATUS_NOT_INITIALIZED) {
            Blasx_Debug_Output("the cublasXtDestroy library was not initialized\n");
        }else if(status==CUBLAS_STATUS_SUCCESS){
            Blasx_Debug_Output("the cublasXt library was successfully destroied\n");
        }
    }
    if (cpublas_handle!=NULL)   {
        int error = dlclose(cpublas_handle);
        if(error == 0) {
            Blasx_Debug_Output("cpublas_handle closed\n");
        }else{
            Blasx_Debug_Output("dlclose error\n");
        }
    }
    if (is_blasx_enable == 1) {
        Blasx_Debug_Output("dest blasx\n");
        blasx_resource_dest( SYS_GPUS, handles_SGEMM, streams_SGEMM, event_SGEMM, C_dev_SGEMM);
        blasx_resource_dest( SYS_GPUS, handles_DGEMM, streams_DGEMM, event_DGEMM, C_dev_DGEMM);
    }
}


