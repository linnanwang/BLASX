#include <blasx_sgemm.h>
cudaError_t    cuda_err;
cublasStatus_t cuda_sta;

void blasx_gpu_sgemm_kernel(int j,
                            int nrowa, int ncola,
                            int nrowb, int ncolb,
                            int nrowc, int ncolc,
                            int current_task, int prior_task,
                            enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                            float* A, float* B, float* C,
                            int lda, int ldb, int ldc,
                            int x, int y, int z,
                            float** C_dev,
                            cudaStream_t *stream, cublasHandle_t *handle_p,
                            int current_stream,
                            float alpha, float beta, int block_dim,
                            int switcher, int* task_batch_counter,
                            LRU_t **LRUs, int GPUs,
                            int *mem_cpy_counter,
                            reader_tracker *addr_track,
                            int GPU_id)
{
    int nrowa_dev, nrowb_dev, nrowc_dev;
    int ncola_dev, ncolb_dev, ncolc_dev;
    int nrow_offset_a, nrow_offset_b;
    int ncol_offset_a, ncol_offset_b;
    int i = current_task/(y+1);
    int k = current_task%(y+1);
    float *A_dev, *B_dev;
    if (TransA == CblasTrans) {
        margin_adjustment(nrowa,ncola,block_dim,j,i,&nrowa_dev,&ncola_dev);
    }else{
        margin_adjustment(nrowa,ncola,block_dim,i,j,&nrowa_dev,&ncola_dev);
    }
    if (TransB == CblasTrans) {
        margin_adjustment(nrowb,ncolb,block_dim,k,j,&nrowb_dev,&ncolb_dev);
    }else{
        margin_adjustment(nrowb,ncolb,block_dim,j,k,&nrowb_dev,&ncolb_dev);
    }
    margin_adjustment(nrowc,ncolc,block_dim,i,k,&nrowc_dev,&ncolc_dev);
    if (TransA == CblasTrans) {
        nrow_offset_a = j*block_dim, ncol_offset_a = i*block_dim;
    }else{
        nrow_offset_a = i*block_dim, ncol_offset_a = j*block_dim;
    }
    if (TransB == CblasTrans) {
        nrow_offset_b = k*block_dim, ncol_offset_b = j*block_dim;
    }else{
        nrow_offset_b = j*block_dim, ncol_offset_b = k*block_dim;
    }
    float *starting_point_A = &A[nrow_offset_a+ncol_offset_a*lda];
    float *starting_point_B = &B[nrow_offset_b+ncol_offset_b*ldb];
    //Asynchonizing set matrix on GPU
    //----------------LRU&RBT optimization----------------//
    mem_control_kernel_float(starting_point_A, &A_dev, LRUs, GPUs, GPU_id, block_dim, mem_cpy_counter, addr_track, stream, nrowa_dev, ncola_dev, lda);
    mem_control_kernel_float(starting_point_B, &B_dev, LRUs, GPUs, GPU_id, block_dim, mem_cpy_counter, addr_track, stream, nrowb_dev, ncolb_dev, ldb);
    //----------------------------------------------------//
    
    if (j == 0) {
        margin_adjustment(nrowc,ncolc,block_dim,i,k,&nrowc_dev,&ncolc_dev);
        int nrow_offset_c = i*block_dim;
        int ncol_offset_c = k*block_dim;
        float *starting_point_C = &C[nrow_offset_c+ncol_offset_c*ldc];
        if (beta != 0) {
            assert( cublasSetMatrixAsync(nrowc_dev, ncolc_dev, sizeof(float), starting_point_C, ldc, C_dev[switcher*STREAMNUM+current_stream], block_dim, *stream) == CUBLAS_STATUS_SUCCESS );
        }
        if (*task_batch_counter != 0) {//Set matrix back
            int i_pre = prior_task/(y+1);
            int k_pre = prior_task%(y+1);
            int nrowc_dev_pre, ncolc_dev_pre;
            margin_adjustment(nrowc,ncolc,block_dim,i_pre,k_pre,&nrowc_dev_pre,&ncolc_dev_pre);
            int nrow_offset_c_pre = i_pre*block_dim;
            int ncol_offset_c_pre = k_pre*block_dim;
            float *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
            assert( cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(float), C_dev[current_stream+(1-switcher)*STREAMNUM], block_dim, starting_point_C_pre, ldc,*stream) == CUBLAS_STATUS_SUCCESS);
        }
    }
    cudaStreamSynchronize(*stream);
    assert( cublasSetStream(*handle_p, *stream) == CUBLAS_STATUS_SUCCESS );

    float beta_inner = (j==0)?(beta):(1);
    int ka = (TransA == CblasTrans)?(nrowa_dev):(ncola_dev);
    cublasOperation_t transa = (TransA == CblasTrans)?(CUBLAS_OP_T):(CUBLAS_OP_N);
    cublasOperation_t transb = (TransB == CblasTrans)?(CUBLAS_OP_T):(CUBLAS_OP_N);
    cublasStatus_t status = cublasSgemm(*handle_p,
                                        transa, transb,
                                        nrowc_dev, ncolc_dev, ka,
                                        &alpha,
                                        A_dev, block_dim,
                                        B_dev, block_dim,
                                        &beta_inner,
                                        C_dev[switcher*STREAMNUM+current_stream], block_dim);
    assert( status == CUBLAS_STATUS_SUCCESS );
}

void blasx_gpu_sgemm(void *arg_data)
{
    int i;
    //----------GPU Argument Prepare------------//
    struct gpu_thread_data *arg = (struct gpu_thread_data *) arg_data;
    const int GPU_id = arg->GPU_id;
    cuda_err = cudaSetDevice(GPU_id);
    assert(cuda_err == cudaSuccess);
    //matrix configuration
    reader_tracker addr_track[1300]; //CRITICAL
    int x                       = arg->mat_conf->x;
    int y                       = arg->mat_conf->y;
    int z                       = arg->mat_conf->z;
    float *A                    = (float*) arg->mat_conf->A;
    float *B                    = (float*) arg->mat_conf->B;
    float *C                    = (float*) arg->mat_conf->C;
    int lda                     = arg->mat_conf->lda;
    int ldb                     = arg->mat_conf->ldb;
    int ldc                     = arg->mat_conf->ldc;
    float beta                  = arg->mat_conf->beta;
    float alpha                 = arg->mat_conf->alpha;
    int nrowa                   = arg->mat_conf->nrowa;
    int nrowb                   = arg->mat_conf->nrowb;
    int nrowc                   = arg->mat_conf->nrowc;
    int ncola                   = arg->mat_conf->ncola;
    int ncolb                   = arg->mat_conf->ncolb;
    int ncolc                   = arg->mat_conf->ncolc;
    enum CBLAS_TRANSPOSE TransA = arg->mat_conf->TransA;
    enum CBLAS_TRANSPOSE TransB = arg->mat_conf->TransB;
    int block_dim               = arg->mat_conf->block_dim;
    //GPU configuration
    const int GPUs              = arg->GPUs;
    LRU_t   **LRUs              = arg->LRUs;
    cublasHandle_t  handle      = handles_SGEMM[GPU_id];
    queue_t *tasks_queue        = arg->tasks_queue;
    //------------------------------------------//
    //hook C_dev
    float *C_dev[STREAMNUM*2];
    for (i = 0; i < STREAMNUM*2; i++) {
        C_dev[i] = C_dev_SGEMM[i+STREAMNUM*GPU_id*2];
    }
    cudaStream_t stream[STREAMNUM];
    cudaEvent_t task_event[STREAMNUM];
    for (i = 0 ; i < STREAMNUM; i++) {
        //hook event
        task_event[i] = event_SGEMM[i+GPU_id*STREAMNUM];
        //hook stream
        stream[i]     = streams_SGEMM[i+GPU_id*STREAMNUM];
    }
    
#ifdef affinity
    //thread setup
    assert( blasx_set_affinity(GPU_id) == 0);
#endif
#ifdef thread_barrier
    pthread_barrier_t* barr     = arg->barr;
    int rc = pthread_barrier_wait(barr);
    assert(!(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD));
#endif
#ifdef thread_profile
    printf("thread%d start@%f\n", GPU_id, get_cur_time());
#endif
    //------------------------------------------//


    //----------------GPU-START-----------------//
    int tasks_rs[STREAMNUM*2]; // mimic reseravation station
    int tasks_rs_size[2] = { 0, 0 };   // always tracking the first unused
    int switcher = 0;
    int task_batch_counter = 0;
    int mem_cpy_counter = 0;

    while (tasks_queue->TAIL >= 0) {
        /*------RS------*/
        int rs_counter          = 0;
        tasks_rs_size[switcher] = 0;
        for (rs_counter = 0; rs_counter < STREAMNUM; rs_counter++) {
            int task_id = dequeue(tasks_queue);
#ifdef task_tracker
            printf("==>GPU%d %d\n", GPU_id, task_id);
#endif
            if (task_id >= 0) {
                tasks_rs[tasks_rs_size[switcher]+STREAMNUM*switcher] = task_id;
                tasks_rs_size[switcher]++;
            }
        }
        
        /*--event_sync---*/
        while (cudaEventQuery(task_event[0]) != cudaSuccess);
        
        /*--reduce_reader--*/
        int addr_counter = 0;
        for (addr_counter = 0; addr_counter < mem_cpy_counter; addr_counter++) {
            void *key          = addr_track[addr_counter].addr;
            int target_GPU_id  = addr_track[addr_counter].GPU_id;
            int is_trans_done  = addr_track[addr_counter].is_trans_done;
            rbt_node *n        = rbt_find(key, &(LRUs[target_GPU_id]->hash_map));
            assert(n != NULL);
            if (is_trans_done == 0 && (target_GPU_id == GPU_id)) {
                assert(target_GPU_id == GPU_id);
                n->associated_LRU_elem->is_trans_done = 1;
            }
            atomic_reader_minus(n);
        }

        /*--kernel_exe---*/
        mem_cpy_counter = 0;
        int j = 0;
        for(j = 0; j <= z; j++){
            for (rs_counter = 0; rs_counter < tasks_rs_size[switcher]; rs_counter++) {
                int current_stream   = rs_counter;
                int current_task   = tasks_rs[rs_counter+STREAMNUM*switcher];
                int prior_task     = tasks_rs[rs_counter+(1-switcher)*STREAMNUM];
                cudaStream_t *curt_stream = &stream[current_stream];
                blasx_gpu_sgemm_kernel(j,
                                       nrowa, ncola,
                                       nrowb, ncolb,
                                       nrowc, ncolc,
                                       current_task, prior_task,
                                       TransA, TransB,
                                       A, B, C,
                                       lda, ldb, ldc,
                                       x, y, z,
                                       C_dev,
                                       curt_stream, &handle,
                                       current_stream,
                                       alpha, beta, block_dim,
                                       switcher, &task_batch_counter,
                                       LRUs, GPUs,
                                       &mem_cpy_counter,
                                       addr_track,
                                       GPU_id);
                if ( j == z && rs_counter == tasks_rs_size[switcher]-1) {
                    /*--event_record--*/
                    cudaError_t err = cudaEventRecord(task_event[0], stream[0]);
                    if(err != cudaSuccess) printf("event record fail\n");
                }
            }
        }
        switcher = 1 - switcher;
        task_batch_counter++;
    }
    //------------------------------------------//

    //---------------RESULT-HARVEST-------------//
    collect_final_result_sgemm(tasks_rs, tasks_rs_size, switcher, stream, C_dev, block_dim, STREAMNUM, x, y, z, nrowc, ncolc, ldc, C);
    //------------------------------------------//
#ifdef thread_profile
    printf("thread%d end@%f\n", GPU_id, get_cur_time());
#endif
}

//dispatch jobs
int blasx_sgemm(const int GPUs, cublasHandle_t* handles, LRU_t **LRUs,
                enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                const int M, const int N, const int K, const float alpha,
                float *A, int lda,
                float *B, int ldb,
                const float beta,
                float *C, int ldc)
{

    //m: the rows of OP(A) and rows C
    //n: the columns of OP(B) and columns C
    //k: the columns of OP(A) and the rows of OP(B)
    /*----Initialization-----*/
    int nrowa, nrowb, nrowc;
    int ncola, ncolb, ncolc;
    int block_dim = BLOCKSIZE_SGEMM;
    
    /*slicing configuration*/
    if (TransA == CblasTrans) {
        nrowa = K;
        ncola = M;
    }else{
        nrowa = M;
        ncola = K;
    }
    if (TransB == CblasTrans) {
        nrowb = N;
        ncolb = K;
    }else{
        nrowb = K;
        ncolb = N;
    }
    nrowc = M;
    ncolc = N;
    int x = (nrowa%block_dim == 0)?(nrowa/block_dim-1):(nrowa/block_dim);
    int y;
    int z = (ncola%block_dim == 0)?(ncola/block_dim-1):(ncola/block_dim);
    if (TransA == CblasTrans) {
        int temp = x;
        x = z;
        z = temp;
    }
    if (TransB == CblasTrans) {
        y = (nrowb%block_dim == 0)?(nrowb/block_dim-1):(nrowb/block_dim);
    }else{
        y = (ncolb%block_dim == 0)?(ncolb/block_dim-1):(ncolb/block_dim);
    }
    
#ifdef thread_barrier
    pthread_barrier_t barr;
    assert( pthread_barrier_init(&barr, NULL, GPUs) == 0 );
#endif
    
    /*---------------GPU-START------------------*/
    int GPU_id = 0;
    pthread_t gpu_tid[GPUs];
    int task_num = (x+1)*(y+1)-1;
    //task queue initlization
    queue_t tasks_queue;
    init_queue(&tasks_queue, task_num);
    struct gpu_thread_data gpu_thread_argument[GPUs];
    // matrix configuartion
    matrix_config      mat_conf;
    mat_conf.A         = A;
    mat_conf.B         = B;
    mat_conf.C         = C;
    mat_conf.nrowa     = nrowa;
    mat_conf.nrowb     = nrowb;
    mat_conf.nrowc     = nrowc;
    mat_conf.ncola     = ncola;
    mat_conf.ncolb     = ncolb;
    mat_conf.ncolc     = ncolc;
    mat_conf.x         = x;
    mat_conf.y         = y;
    mat_conf.z         = z;
    mat_conf.block_dim = block_dim;
    mat_conf.alpha     = alpha;
    mat_conf.beta      = beta;
    mat_conf.lda       = lda;
    mat_conf.ldb       = ldb;
    mat_conf.ldc       = ldc;
    mat_conf.TransA    = TransA;
    mat_conf.TransB    = TransB;

    //-------------------Execute-----------------//
    for (GPU_id = 0 ; GPU_id < GPUs ; GPU_id++){
        // GPU settings
        gpu_thread_argument[GPU_id].GPUs        = GPUs;
        gpu_thread_argument[GPU_id].LRUs        = LRUs;
        gpu_thread_argument[GPU_id].GPU_id      = GPU_id;
        gpu_thread_argument[GPU_id].handles     = handles;
        gpu_thread_argument[GPU_id].mat_conf    = &mat_conf;
        gpu_thread_argument[GPU_id].tasks_queue = &tasks_queue;
#ifdef thread_barrier
        gpu_thread_argument[GPU_id].barr        = &barr;
#endif
        int err=pthread_create(&gpu_tid[GPU_id], NULL, (void *)&blasx_gpu_sgemm, (void *)&gpu_thread_argument[GPU_id]);
        if (err != 0) printf("\ncan't create thread ");
    }
    
    /*-------------ALL-THREADS-MERGE------------*/
    for (GPU_id = 0; GPU_id < GPUs; GPU_id++){
        pthread_join(gpu_tid[GPU_id], NULL);
    }
    return EXIT_SUCCESS;
}
