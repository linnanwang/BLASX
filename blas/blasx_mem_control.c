#include <blasx_mem_control.h>

void blasx_get_index(int tile_index, int row_start, int row_end, int* x, int* y, enum CBLAS_UPLO uplo, int ROWs) {
    if(row_end < row_start) {
        return;
    }
    int m = (row_end + row_start)/2;
    if(m*(1+m)/2 <= tile_index && (m+1)*(m+2)/2 > tile_index) {
        if (uplo == CblasLower) {
            *x = m;
            *y = tile_index - (m)*(1+m)/2;
        } else {
            *x = ROWs - m;
            *y = ROWs - tile_index + (m)*(1+m)/2;
        }
        return;
    } else if(m*(1+m)/2 >= tile_index && (m+1)*(m+2)/2 >= tile_index) {
        blasx_get_index(tile_index, row_start, m-1, x, y, uplo, ROWs);
    } else if(m*(1+m)/2 <= tile_index && (m+1)*(m+2)/2 <= tile_index) {
        blasx_get_index(tile_index, m+1, row_end, x, y, uplo, ROWs);
    }
}

void margin_adjustment(int row_original_matrix, int col_original_matrix, int block_dim, int block_row, int block_col, int *nrow_dev, int *ncol_dev) {
    if (block_row == row_original_matrix/block_dim && block_col != col_original_matrix/block_dim) {
        int row_residual = row_original_matrix % block_dim;
        *nrow_dev = row_residual;
        *ncol_dev = block_dim;
    }else if(block_row != row_original_matrix/block_dim && block_col == col_original_matrix/block_dim){
        int col_residual = col_original_matrix % block_dim;
        *nrow_dev = block_dim;
        *ncol_dev = col_residual;
    }else if(block_row == row_original_matrix/block_dim && block_col == col_original_matrix/block_dim){
        int row_residual = row_original_matrix % block_dim;
        int col_residual = col_original_matrix % block_dim;
        *nrow_dev = row_residual;
        *ncol_dev = col_residual;
    }else{
        *nrow_dev = block_dim;
        *ncol_dev = block_dim;
    }
}


void atomic_reader_plus(volatile rbt_node* node) {
    assert(node != NULL);
    while (1) {
        int old_read_tracker = (node->associated_LRU_elem->read_tracker);
        if ( __sync_bool_compare_and_swap(&(node->associated_LRU_elem->read_tracker), old_read_tracker, old_read_tracker+1) ) {
            return;
        }
    }
}

void atomic_reader_minus(volatile rbt_node* node) {
    assert(node != NULL);
    while (1) {
        int old_read_tracker = (node->associated_LRU_elem->read_tracker);
        if ( __sync_bool_compare_and_swap(&(node->associated_LRU_elem->read_tracker), old_read_tracker, old_read_tracker-1) ) {
            return;
        }
    }
}

void collect_final_result_sgemm(int *tasks_rs, int *tasks_rs_size, int switcher, cudaStream_t *stream, float** C_dev, int block_dim, int stream_num, int x, int y, int z, int nrowc, int ncolc, int ldc, float *C) {
    switcher = 1 - switcher;
    int temp = 0;
    for (temp = tasks_rs_size[switcher]; temp < tasks_rs_size[1-switcher] ; temp++) {
        int prior_task = tasks_rs[temp+stream_num*(1-switcher)];
        int i_pre = prior_task/(y+1);
        int k_pre = prior_task%(y+1);
        int current_stream = temp;
        int nrowc_dev_pre, ncolc_dev_pre;
        margin_adjustment(nrowc, ncolc, block_dim, i_pre, k_pre, &nrowc_dev_pre, &ncolc_dev_pre);
        int nrow_offset_c_pre = i_pre*block_dim;
        int ncol_offset_c_pre = k_pre*block_dim;
        float *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
        assert( cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(float), C_dev[current_stream+(1-switcher)*stream_num], block_dim, starting_point_C_pre, ldc,stream[current_stream]) == CUBLAS_STATUS_SUCCESS );
        assert(cudaStreamSynchronize(stream[current_stream]) == cudaSuccess);
    }
    for (temp = 0; temp < tasks_rs_size[switcher]; temp++) {
        int prior_task = tasks_rs[temp+stream_num*(switcher)];
        int i_pre = prior_task/(y+1);
        int k_pre = prior_task%(y+1);
        int current_stream = temp;
        int nrowc_dev_pre, ncolc_dev_pre;
        margin_adjustment(nrowc, ncolc, block_dim, i_pre, k_pre, &nrowc_dev_pre, &ncolc_dev_pre);
        int nrow_offset_c_pre = i_pre*block_dim;
        int ncol_offset_c_pre = k_pre*block_dim;
        float *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
        assert( cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(float), C_dev[current_stream+switcher*stream_num], block_dim, starting_point_C_pre, ldc,stream[current_stream]) == CUBLAS_STATUS_SUCCESS );
        assert(cudaStreamSynchronize(stream[current_stream]) == cudaSuccess);
    }
}

void collect_final_result_dgemm(int *tasks_rs, int *tasks_rs_size, int switcher, cudaStream_t *stream, double** C_dev, int block_dim, int stream_num, int x, int y, int z, int nrowc, int ncolc, int ldc, double *C) {
    switcher = 1 - switcher;
    int temp = 0;
    for (temp = tasks_rs_size[switcher]; temp < tasks_rs_size[1-switcher] ; temp++) {
        int prior_task = tasks_rs[temp+stream_num*(1-switcher)];
        int i_pre = prior_task/(y+1);
        int k_pre = prior_task%(y+1);
        int current_stream = temp;
        int nrowc_dev_pre, ncolc_dev_pre;
        margin_adjustment(nrowc, ncolc, block_dim, i_pre, k_pre, &nrowc_dev_pre, &ncolc_dev_pre);
        int nrow_offset_c_pre = i_pre*block_dim;
        int ncol_offset_c_pre = k_pre*block_dim;
        double *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
        assert( cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(double), C_dev[current_stream+(1-switcher)*stream_num], block_dim, starting_point_C_pre, ldc,stream[current_stream]) == CUBLAS_STATUS_SUCCESS );
        assert(cudaStreamSynchronize(stream[current_stream]) == cudaSuccess);
    }
    for (temp = 0; temp < tasks_rs_size[switcher]; temp++) {
        int prior_task = tasks_rs[temp+stream_num*(switcher)];
        int i_pre = prior_task/(y+1);
        int k_pre = prior_task%(y+1);
        int current_stream = temp;
        int nrowc_dev_pre, ncolc_dev_pre;
        margin_adjustment(nrowc, ncolc, block_dim, i_pre, k_pre, &nrowc_dev_pre, &ncolc_dev_pre);
        int nrow_offset_c_pre = i_pre*block_dim;
        int ncol_offset_c_pre = k_pre*block_dim;
        double *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
        assert( cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(double), C_dev[current_stream+switcher*stream_num], block_dim, starting_point_C_pre, ldc,stream[current_stream]) == CUBLAS_STATUS_SUCCESS );
        assert(cudaStreamSynchronize(stream[current_stream]) == cudaSuccess);
    }
}

void collect_final_result_dtrsm_mode_1(int *tasks_rs, int *tasks_rs_size, int switcher, int switcher_rs, cudaStream_t *stream, double** buffer_dev, int block_dim, int stream_num, int x, int y, int z, int nrowb, int ncolb, int ldb, double *B, int* switch_tracker) {
    int temp = 0;
    for (temp = tasks_rs_size[switcher_rs]; temp < tasks_rs_size[1-switcher_rs] ; temp++) {
        //            printf("retrieving B[%d, %d] @stream=%d switcher:%d\n", z, tasks_rs[temp+STREAMNUM*(1-switcher_rs)], temp, switcher);
        int row = z;
        int col = tasks_rs[temp+stream_num*(1-switcher_rs)];
        int current_stream = temp;
        int nrow_offset = row*block_dim;
        int ncol_offset = col*block_dim;
        int nrow_dev, ncol_dev;
        margin_adjustment(nrowb, ncolb, block_dim, row, col, &nrow_dev, &ncol_dev);
        double *starting_point = &B[nrow_offset+ncol_offset*ldb];
        cublasGetMatrixAsync(nrow_dev, ncol_dev, sizeof(double), buffer_dev[current_stream+switch_tracker[temp]*stream_num], block_dim, starting_point, ldb, stream[current_stream]);
        cudaStreamSynchronize(stream[current_stream]);
    }
    for (temp = 0; temp < tasks_rs_size[switcher_rs]; temp++) {
        //assume 1-switcher
        //printf("retrieving B[%d, %d] @stream=%d\n", z, tasks_rs[temp+STREAMNUM*switcher_rs], temp);
        int row = z;
        int col = tasks_rs[temp+stream_num*switcher_rs];
        int current_stream = temp;
        int nrow_offset = row*block_dim;
        int ncol_offset = col*block_dim;
        int nrow_dev, ncol_dev;
        margin_adjustment(nrowb, ncolb, block_dim, row, col, &nrow_dev, &ncol_dev);
        double *starting_point = &B[nrow_offset+ncol_offset*ldb];
        cublasGetMatrixAsync(nrow_dev, ncol_dev, sizeof(double), buffer_dev[current_stream+switch_tracker[temp]*stream_num], block_dim, starting_point, ldb, stream[current_stream]);
        cudaStreamSynchronize(stream[current_stream]);
    }
}

void collect_final_result_dsyrk_syr2k(int *tasks_rs, int *tasks_rs_size, int switcher, cudaStream_t *stream, double** C_dev, int block_dim, int stream_num, int x,int y, int z, int nrowc, int ncolc, int ldc, double *C, enum CBLAS_UPLO Uplo) {
    switcher = 1 - switcher;
    int temp = 0;
    for (temp = tasks_rs_size[switcher]; temp < tasks_rs_size[1-switcher] ; temp++) {
        int prior_task = tasks_rs[temp+stream_num*(1-switcher)];
        int i_pre, k_pre;
        blasx_get_index(prior_task, 0, x, &i_pre, &k_pre, Uplo, x);
        int current_stream = temp;
        int nrowc_dev_pre, ncolc_dev_pre;
        margin_adjustment(nrowc, ncolc, block_dim, i_pre, k_pre, &nrowc_dev_pre, &ncolc_dev_pre);
        int nrow_offset_c_pre = i_pre*block_dim;
        int ncol_offset_c_pre = k_pre*block_dim;
        double *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
        cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(double), C_dev[current_stream+(1-switcher)*stream_num], block_dim, starting_point_C_pre, ldc,stream[current_stream]);
        cudaStreamSynchronize(stream[current_stream]);
    }
    for (temp = 0; temp < tasks_rs_size[switcher]; temp++) {
        //assume 1-switcher
        int prior_task = tasks_rs[temp+stream_num*(switcher)];
        int i_pre, k_pre;
        blasx_get_index(prior_task, 0, x, &i_pre, &k_pre, Uplo, x);
        int current_stream = temp;
        int nrowc_dev_pre, ncolc_dev_pre;
        margin_adjustment(nrowc, ncolc, block_dim, i_pre, k_pre, &nrowc_dev_pre, &ncolc_dev_pre);
        int nrow_offset_c_pre = i_pre*block_dim;
        int ncol_offset_c_pre = k_pre*block_dim;
        double *starting_point_C_pre = &C[nrow_offset_c_pre+ncol_offset_c_pre*ldc];
        cublasGetMatrixAsync(nrowc_dev_pre, ncolc_dev_pre, sizeof(double), C_dev[current_stream+switcher*stream_num], block_dim, starting_point_C_pre, ldc,stream[current_stream]);
        cudaStreamSynchronize(stream[current_stream]);
    }
}


void mem_control_kernel_double(double *starting_point_A, double **A_dev,
                        LRU_t **LRUs, const int GPUs, const int GPU_id, int block_dim,
                        int *mem_cpy_counter, reader_tracker *addr_track,
                        cudaStream_t *stream,
                        int nrowa_dev, int ncola_dev, int lda) {
    rbt_node* block_A = rbt_find(starting_point_A, &(LRUs[GPU_id]->hash_map));
    if( block_A == NULL ) { //new element
        //fprintf(stderr, "==========new element========\n");
        //traverse_LRU_se(LRU);
        int search_l_GPU = GPU_id-1;
        int search_r_GPU = GPU_id+1;
        rbt_node *block_A_l = NULL;
        rbt_node *block_A_r = NULL;
        while (block_A_l == NULL && block_A_r == NULL) {
            if (search_l_GPU >= 0) {
                block_A_l = rbt_find(starting_point_A, &(LRUs[search_l_GPU]->hash_map));
                if (block_A_l != NULL) {
                    if (block_A_l->associated_LRU_elem->is_trans_done == 0) {
                        int peer_access_check = 0;
                        cudaDeviceCanAccessPeer(&peer_access_check, GPU_id, search_l_GPU);
                        if(peer_access_check == 1) block_A_l = NULL;
                    }
                }
                search_l_GPU--;
            }
            if (search_r_GPU < GPUs) {
                block_A_r = rbt_find(starting_point_A, &(LRUs[search_r_GPU]->hash_map));
                if (block_A_r != NULL) {
                    if (block_A_r->associated_LRU_elem->is_trans_done == 0) {
                        int peer_access_check = 0;
                        cudaDeviceCanAccessPeer(&peer_access_check, GPU_id, search_r_GPU);
                        if(peer_access_check == 1) block_A_r = NULL;
                    }
                }
                search_r_GPU++;
            }
            if (search_l_GPU < 0 && search_r_GPU >= GPUs) {
                break;
            }
        }
        //rectitfication
        search_l_GPU++; search_r_GPU--;
        assert(search_l_GPU >= 0 && search_l_GPU < GPUs);
        assert(search_r_GPU >= 0 && search_r_GPU < GPUs);
        
        if ( !(block_A_l == NULL && block_A_r == NULL) ) {
            //inter GPU communication
            int target_GPU_id = 0;
            if (block_A_l != NULL && block_A_r != NULL) {
                if (ABS(search_l_GPU - GPU_id) > ABS(search_r_GPU - GPU_id)) {
                    target_GPU_id = search_r_GPU;
                    block_A       = block_A_r;
                } else if(ABS(search_l_GPU - GPU_id) < ABS(search_r_GPU - GPU_id)) {
                    target_GPU_id = search_l_GPU;
                    block_A       = block_A_l;
                } else {
                    int rand_select = rand()%10;
                    if (rand_select < 5) {
                        target_GPU_id = search_l_GPU;
                        block_A       = block_A_l;
                    } else {
                        target_GPU_id = search_r_GPU;
                        block_A       = block_A_r;
                    }
                }
                if(block_A->associated_LRU_elem->is_trans_done != 1)
                goto new_block;
                //fprintf(stderr, "==>3  block on GPUs:(%d, %d), but chose %d(done:%d) as curt GPU is %d (block_A_l:%p, block_A_r:%p)\n", search_l_GPU, search_r_GPU, target_GPU_id, block_A->associated_LRU_elem->is_trans_done, GPU_id, block_A_l, block_A_r);
            } else {
                if (block_A_l != NULL && block_A_r == NULL) {
                    target_GPU_id = search_l_GPU;
                    block_A       = block_A_l;
                } else if(block_A_r != NULL && block_A_l == NULL) {
                    target_GPU_id = search_r_GPU;
                    block_A       = block_A_r;
                }
                if(block_A->associated_LRU_elem->is_trans_done != 1)
                goto new_block;
                //printf("==>2  block on GPUs:%d, and curt GPU is %d (done:%d)\n", target_GPU_id, GPU_id, block_A->associated_LRU_elem->is_trans_done);
            }
            if (rbt_find(starting_point_A, &(LRUs[target_GPU_id]->hash_map)) == NULL)
            goto new_block;
            atomic_reader_plus(block_A);
            *A_dev = (double*) LRU_in(starting_point_A, LRUs[GPU_id], sizeof(double)*block_dim*block_dim, GPU_id);
            assert( rbt_find(starting_point_A, &(LRUs[target_GPU_id]->hash_map)) != NULL);
            assert( rbt_find(starting_point_A, &(LRUs[target_GPU_id]->hash_map))->associated_LRU_elem->is_trans_done == 1);
            assert( cudaMemcpyPeerAsync(*A_dev, GPU_id, block_A->associated_LRU_elem->GPU_p, target_GPU_id, sizeof(double)*block_dim*block_dim, *stream) == cudaSuccess );
            //cannot dequeue the GPU mem at the target GPU
            addr_track[*mem_cpy_counter].addr   = starting_point_A;
            addr_track[*mem_cpy_counter].GPU_id = target_GPU_id;
            addr_track[*mem_cpy_counter].is_trans_done = 1;
            (*mem_cpy_counter) += 1;
            //cannnot dequeue the current new GPU mem
            addr_track[*mem_cpy_counter].addr   = starting_point_A;
            addr_track[*mem_cpy_counter].GPU_id = GPU_id;
            addr_track[*mem_cpy_counter].is_trans_done = 0;
            (*mem_cpy_counter) += 1;
        } else {
        new_block:
            //(block_A_r == NULL && block_A_l == NULL) {
            //bring new blocks
            //printf("==>1  bring new block to GPU:%d\n", GPU_id);
            (*A_dev) = (double*) LRU_in(starting_point_A, LRUs[GPU_id], sizeof(double)*block_dim*block_dim, GPU_id);
            assert( cublasSetMatrixAsync(nrowa_dev, ncola_dev, sizeof(double), starting_point_A, lda, *A_dev, block_dim, *stream) == CUBLAS_STATUS_SUCCESS );
            addr_track[*mem_cpy_counter].addr          = starting_point_A;
            addr_track[*mem_cpy_counter].GPU_id        = GPU_id;
            addr_track[*mem_cpy_counter].is_trans_done = 0;
            (*mem_cpy_counter) += 1;
        }
    } else {
        atomic_reader_plus(block_A);
        assert( rbt_find(starting_point_A, &(LRUs[GPU_id]->hash_map)) != NULL);
        *A_dev = (double*) LRU_reorder(starting_point_A, LRUs[GPU_id]);
        addr_track[*mem_cpy_counter].addr   = starting_point_A;
        addr_track[*mem_cpy_counter].GPU_id = GPU_id;
        (*mem_cpy_counter) += 1;
    }

}

void mem_control_kernel_float(float *starting_point_A, float **A_dev,
                        LRU_t **LRUs, const int GPUs, const int GPU_id, int block_dim,
                        int *mem_cpy_counter, reader_tracker *addr_track,
                        cudaStream_t *stream,
                        int nrowa_dev, int ncola_dev, int lda) {
    rbt_node* block_A = rbt_find(starting_point_A, &(LRUs[GPU_id]->hash_map));
    if( block_A == NULL ) { //new element
        //fprintf(stderr, "==========new element========\n");
        //traverse_LRU_se(LRU);
        int search_l_GPU = GPU_id-1;
        int search_r_GPU = GPU_id+1;
        rbt_node *block_A_l = NULL;
        rbt_node *block_A_r = NULL;
        while (block_A_l == NULL && block_A_r == NULL) {
            if (search_l_GPU >= 0) {
                block_A_l = rbt_find(starting_point_A, &(LRUs[search_l_GPU]->hash_map));
                if (block_A_l != NULL) {
                    if (block_A_l->associated_LRU_elem->is_trans_done == 0) {
                        int peer_access_check = 0;
                        cudaDeviceCanAccessPeer(&peer_access_check, GPU_id, search_l_GPU);
                        if(peer_access_check == 1) block_A_l = NULL;
                    }
                }
                search_l_GPU--;
            }
            if (search_r_GPU < GPUs) {
                block_A_r = rbt_find(starting_point_A, &(LRUs[search_r_GPU]->hash_map));
                if (block_A_r != NULL) {
                    if (block_A_r->associated_LRU_elem->is_trans_done == 0) {
                        int peer_access_check = 0;
                        cudaDeviceCanAccessPeer(&peer_access_check, GPU_id, search_r_GPU);
                        if(peer_access_check == 1) block_A_r = NULL;
                    }
                }
                search_r_GPU++;
            }
            if (search_l_GPU < 0 && search_r_GPU >= GPUs) {
                break;
            }
        }
        //rectitfication
        search_l_GPU++; search_r_GPU--;
        assert(search_l_GPU >= 0 && search_l_GPU < GPUs);
        assert(search_r_GPU >= 0 && search_r_GPU < GPUs);
        
        if ( !(block_A_l == NULL && block_A_r == NULL) ) {
            //inter GPU communication
            int target_GPU_id = 0;
            if (block_A_l != NULL && block_A_r != NULL) {
                if (ABS(search_l_GPU - GPU_id) > ABS(search_r_GPU - GPU_id)) {
                    target_GPU_id = search_r_GPU;
                    block_A       = block_A_r;
                } else if(ABS(search_l_GPU - GPU_id) < ABS(search_r_GPU - GPU_id)) {
                    target_GPU_id = search_l_GPU;
                    block_A       = block_A_l;
                } else {
                    int rand_select = rand()%10;
                    if (rand_select < 5) {
                        target_GPU_id = search_l_GPU;
                        block_A       = block_A_l;
                    } else {
                        target_GPU_id = search_r_GPU;
                        block_A       = block_A_r;
                    }
                }
                if(block_A->associated_LRU_elem->is_trans_done != 1)
                goto new_block;
                //fprintf(stderr, "==>3  block on GPUs:(%d, %d), but chose %d(done:%d) as curt GPU is %d (block_A_l:%p, block_A_r:%p)\n", search_l_GPU, search_r_GPU, target_GPU_id, block_A->associated_LRU_elem->is_trans_done, GPU_id, block_A_l, block_A_r);
            } else {
                if (block_A_l != NULL && block_A_r == NULL) {
                    target_GPU_id = search_l_GPU;
                    block_A       = block_A_l;
                } else if(block_A_r != NULL && block_A_l == NULL) {
                    target_GPU_id = search_r_GPU;
                    block_A       = block_A_r;
                }
                if(block_A->associated_LRU_elem->is_trans_done != 1)
                goto new_block;
                //printf("==>2  block on GPUs:%d, and curt GPU is %d (done:%d)\n", target_GPU_id, GPU_id, block_A->associated_LRU_elem->is_trans_done);
            }
            if (rbt_find(starting_point_A, &(LRUs[target_GPU_id]->hash_map)) == NULL)
            goto new_block;
            atomic_reader_plus(block_A);
            *A_dev = (float*) LRU_in(starting_point_A, LRUs[GPU_id], sizeof(float)*block_dim*block_dim, GPU_id);
            assert( rbt_find(starting_point_A, &(LRUs[target_GPU_id]->hash_map)) != NULL);
            assert( rbt_find(starting_point_A, &(LRUs[target_GPU_id]->hash_map))->associated_LRU_elem->is_trans_done == 1);
            assert( cudaMemcpyPeerAsync(*A_dev, GPU_id, block_A->associated_LRU_elem->GPU_p, target_GPU_id, sizeof(float)*block_dim*block_dim, *stream) == cudaSuccess );
            //cannot dequeue the GPU mem at the target GPU
            addr_track[*mem_cpy_counter].addr   = starting_point_A;
            addr_track[*mem_cpy_counter].GPU_id = target_GPU_id;
            addr_track[*mem_cpy_counter].is_trans_done = 1;
            (*mem_cpy_counter) += 1;
            //cannnot dequeue the current new GPU mem
            addr_track[*mem_cpy_counter].addr   = starting_point_A;
            addr_track[*mem_cpy_counter].GPU_id = GPU_id;
            addr_track[*mem_cpy_counter].is_trans_done = 0;
            (*mem_cpy_counter) += 1;
        } else {
        new_block:
            //(block_A_r == NULL && block_A_l == NULL) {
            //bring new blocks
            //printf("==>1  bring new block to GPU:%d\n", GPU_id);
            (*A_dev) = (float*) LRU_in(starting_point_A, LRUs[GPU_id], sizeof(float)*block_dim*block_dim, GPU_id);
            assert( cublasSetMatrixAsync(nrowa_dev, ncola_dev, sizeof(float), starting_point_A, lda, *A_dev, block_dim, *stream) == CUBLAS_STATUS_SUCCESS );
            addr_track[*mem_cpy_counter].addr          = starting_point_A;
            addr_track[*mem_cpy_counter].GPU_id        = GPU_id;
            addr_track[*mem_cpy_counter].is_trans_done = 0;
            (*mem_cpy_counter) += 1;
        }
    } else {
        atomic_reader_plus(block_A);
        assert( rbt_find(starting_point_A, &(LRUs[GPU_id]->hash_map)) != NULL);
        *A_dev = (float*) LRU_reorder(starting_point_A, LRUs[GPU_id]);
        addr_track[*mem_cpy_counter].addr   = starting_point_A;
        addr_track[*mem_cpy_counter].GPU_id = GPU_id;
        (*mem_cpy_counter) += 1;
    }
}
