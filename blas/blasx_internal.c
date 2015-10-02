#include <blasx_internal.h>
/****************************************/
int deviceID[32] = {0};
/*****************Initialization***********************/

double get_cur_time() {
    struct timeval   tv;
    struct timezone  tz;
    double cur_time;
    gettimeofday(&tv, &tz);
    cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
    return cur_time;
}

inline void Blasx_Debug_Output(const char *fmt, ...)
{
#if defined BLASX_DEBUG
    fprintf(stderr,"BlasX Debug Info--->  ");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif /* BLASX_DEBUG */
}

void cpublas_init(){
    blasx_mutex_lock();
    assert( cpublas_handle == NULL );
    Blasx_Debug_Output("dlopen the cpublas:%p\n",cpublas_handle);
    cpublas_handle = dlopen(blas_path, RTLD_LAZY);
    if (!cpublas_handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    blasx_mutex_unlock();
}

void cublasxt_init(){
    blasx_mutex_lock();
    cublasStatus_t status;
    assert( cublasXt_handle == NULL );
    Blasx_Debug_Output("    Creating cublasXt_handle\n");
    assert( cublasXtCreate(&cublasXt_handle) == CUBLAS_STATUS_SUCCESS );
    Blasx_Debug_Output("    Selecting cublasXt_device\n");
    assert( cublasXtDeviceSelect(cublasXt_handle, 1, deviceID) == CUBLAS_STATUS_SUCCESS );
    Blasx_Debug_Output("    Setting PinningMemMode for cublasXt\n");
    assert( cublasXtSetPinningMemMode(cublasXt_handle, CUBLASXT_PINNING_ENABLED) == CUBLAS_STATUS_SUCCESS );
    blasx_mutex_unlock();
}

void blasx_resource_init(int GPUs, cublasHandle_t* handles, cudaStream_t* streams, cudaEvent_t* events, void** C_dev, int is_float) {
    if(is_float == 1) C_dev = (float**)  C_dev;
    else              C_dev = (double**) C_dev;
    int GPU_id = 0;
    for (GPU_id = 0; GPU_id < GPUs; GPU_id++) {
        assert( cudaSetDevice(GPU_id) == cudaSuccess );
        //create handles
        assert( cublasCreate(&handles[GPU_id]) == CUBLAS_STATUS_SUCCESS);
        //create streams and event
        int i = 0;
        for (i = 0 ; i < STREAMNUM; i++) {
            assert( cudaStreamCreate(&streams[i+GPU_id*STREAMNUM]) == cudaSuccess );
            assert( cudaEventCreateWithFlags(&events[i+GPU_id*STREAMNUM], cudaEventDisableTiming) == cudaSuccess );
        }
        //create C_dev
        for (i = 0; i < STREAMNUM*2; i++) {
            if (is_float == 1) {
                assert( cudaMalloc((void**)&C_dev[i+GPU_id*STREAMNUM*2], sizeof(float)*BLOCKSIZE_SGEMM*BLOCKSIZE_SGEMM) == cudaSuccess );
            } else {
                 assert( cudaMalloc((void**)&C_dev[i+GPU_id*STREAMNUM*2], sizeof(double)*BLOCKSIZE_DGEMM*BLOCKSIZE_DGEMM) == cudaSuccess );
            }
        }
    }
}

void blasx_tile_init(){
    blasx_mutex_lock();
    cudaGetDeviceCount(&SYS_GPUS);
    assert(SYS_GPUS != 0);
    //GEMM
    blasx_resource_init( SYS_GPUS, handles_SGEMM, streams_SGEMM, event_SGEMM, C_dev_SGEMM, 1);
    blasx_resource_init( SYS_GPUS, handles_DGEMM, streams_DGEMM, event_DGEMM, C_dev_DGEMM, 0);
    //flag init
    is_blasx_enable = 1;
    blasx_mutex_unlock();
}

void blasx_init(const blasx_operation_t t){
    switch (t) {
        case CPU:            cpublas_init();       break;
        case CUBLASXT:       cublasxt_init();      break;
        case BLASX:          blasx_tile_init();    break;
        default:
            break;
    }
}

/*----input format transformation among blas, cublas, cublasXt------*/
inline int CBLasTransToCuBlasTrans(enum CBLAS_TRANSPOSE transa, cublasOperation_t *t){
    switch (transa) {
        case CblasConjTrans: *t=CUBLAS_OP_C; return 0;
        case CblasTrans: *t=CUBLAS_OP_T; return 0;
        case CblasNoTrans: *t=CUBLAS_OP_N; return 0;
        default:    return -1;
    }
}

inline int CBlasSideToCuBlasSide(enum CBLAS_SIDE side,cublasSideMode_t *t){
    switch (side) {
        case CblasLeft: *t=CUBLAS_SIDE_LEFT; return 0;
        case CblasRight: *t=CUBLAS_SIDE_RIGHT; return 0;
        default:    return -1;
    }
}

inline int CBlasDiagModeToCuBlasDiagMode(enum CBLAS_DIAG diag, cublasDiagType_t *t){
    switch (diag) {
        case CblasNonUnit: *t=CUBLAS_DIAG_NON_UNIT; return 0;
        case CblasUnit: *t=CUBLAS_DIAG_UNIT; return 0;
        default:    return -1;
    }
}

inline int CBlasFilledModeToCuBlasFilledMode(enum CBLAS_UPLO uplo, cublasFillMode_t *t){
    switch (uplo) {
        case CblasUpper: *t=CUBLAS_FILL_MODE_UPPER; return 0;
        case CblasLower: *t=CUBLAS_FILL_MODE_LOWER; return 0;
        default:    return -1;
    }
}

inline int F77UploToCBlasUplo(char *type, enum CBLAS_UPLO *uplo) {
    if(*type == 'U'|| *type == 'u') {
        *uplo=CblasUpper;
        return 0;
    }else if(*type == 'L'|| *type=='l'){
        *uplo=CblasLower;
        return 0;
    }else{
        return -1;
    }
}

inline int F77SideToCBlasSide(char *type, enum CBLAS_SIDE *side) {
    if(*type == 'R'|| *type == 'r'){
        *side=CblasRight;
        return 0;
    }else if(*type == 'L'|| *type=='l'){
        *side=CblasLeft;
        return 0;
    }
    else{
        return -1;
    }
}

inline int F77TransToCBLASTrans(char *type, enum CBLAS_TRANSPOSE *trans) {
    if(*type == 'N'|| *type == 'n'){
        *trans=CblasNoTrans;
        return 0;
    }else if(*type == 'T'|| *type=='t'){
        *trans=CblasTrans;
        return 0;
    }else if(*type == 'C'|| *type=='c'){
        *trans=CblasConjTrans;
        return 0;
    }else{
        return -1;
    }
}

inline int F77DiagToCBLASDiag(char *type, enum CBLAS_DIAG *diag) {
    if(*type == 'N'|| *type == 'n') {
        *diag=CblasNonUnit;
        return 0;
    }else if(*type == 'U'|| *type=='u'){
        *diag=CblasUnit;
        return 0;
    }else{
        return -1;
    }
}












