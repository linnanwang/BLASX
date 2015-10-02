#include <LRU.h>
//#define BLASX_MALLOC

cudaError_t    cuda_err;

void traverse_LRU_se(const LRU_t *LRU ) {
    fprintf(stderr, "traverse: a  ");
    int counter = 0;
    LRU_elem *s = LRU->head;
    while (s != NULL) {
        counter++;
        fprintf(stderr, "(%p->%p,%d, is_done:%d)->", s, s->key, s->read_tracker, s->is_trans_done);
        s = s->next;
    }
    printf("\n");
}

void traverse_LRU_es(const LRU_t *LRU ) {
    printf("traverse:   ");
    LRU_elem *e = LRU->tail;
    while (e != NULL) {
        printf("(%p,%d)->", e->key, e->read_tracker);
        e = e->prior;
    }
    printf("\n");
}

LRU_t* LRU_init(int GPU_id) {
    LRU_t *LRU      = (LRU_t*) malloc(sizeof(LRU_t));
    LRU->hash_map   = rbt_init();
#ifdef BLASX_MALLOC
    LRU->gpu_mem    = blasx_gpu_malloc_init(GPU_id);
#endif
    LRU->head       = NULL;
    LRU->tail       = NULL;
    return LRU;
}

void LRU_free(LRU_t* LRU, int GPU_id) {
#ifdef BLASX_MALLOC
    cuda_err = cudaSetDevice(GPU_id);
    assert(cuda_err == cudaSuccess);
    blasx_gpu_malloc_fini(LRU->gpu_mem, GPU_id);
#endif
    rbt_free(&(LRU->hash_map));
    LRU_elem* s = LRU->head;
    int counter = 0;
    while (s != NULL) {
        counter++;
        LRU_elem* to_delete = s;
        s = s->next;
#ifndef BLASX_MALLOC
        cudaFree(to_delete->GPU_p);
#endif
        to_delete->key = NULL;
        to_delete->GPU_p = NULL;
        to_delete->prior = NULL;
        to_delete->next = NULL;
        to_delete->read_tracker = 0;
        to_delete->is_trans_done = 0;
        free(to_delete);
    }
    free(LRU);
}

T LRU_reorder(T key, LRU_t* LRU) {
    assert(rbt_find(key, &(LRU->hash_map)) != NULL);
    LRU_elem *old_head      = LRU->head;
    LRU_elem *deranked_elem = rbt_find(key, &(LRU->hash_map))->associated_LRU_elem;
    if (deranked_elem->prior != NULL && deranked_elem->next != NULL) {
        LRU_elem *deranked_prior = deranked_elem->prior;
        LRU_elem *deranked_next  = deranked_elem->next;
        deranked_prior->next     = deranked_next;
        deranked_next->prior     = deranked_prior;
        deranked_elem->prior     = NULL;
        deranked_elem->next      = old_head;
        LRU->head                = deranked_elem;
        deranked_elem->next      = old_head;
        deranked_elem->prior     = NULL;
        old_head->prior          = deranked_elem;
    } else if(deranked_elem->prior != NULL && deranked_elem->next == NULL) {
        LRU_elem *deranked_prior = deranked_elem->prior;
        deranked_prior->next     = NULL;
        deranked_elem->prior     = NULL;
        deranked_elem->next      = old_head;
        old_head->prior          = deranked_elem;
        LRU->head                = deranked_elem;
        LRU->tail                = deranked_prior;
    }
    return deranked_elem->GPU_p;
}

// return the corresponding GPU pointer
// return the GPU pointer, and input the CPU pointer
T LRU_in(T key, LRU_t* LRU, size_t mem_size, int GPU_id) {
    //printf("new element key:%p ", key);
    LRU_elem *new_head = (LRU_elem*) malloc(sizeof(LRU_elem));
    //printf("LRU block: %p ", new_head);
    new_head->key           = key;
    new_head->prior         = NULL;
    new_head->next          = NULL;
    new_head->read_tracker  = 1;
    new_head->is_trans_done = 0;
    void *new_gpu_block = NULL;
    while (new_gpu_block == NULL) {
        T outkey        = NULL;
#ifdef BLASX_MALLOC
        new_gpu_block   = (void*) blasx_gpu_malloc(LRU->gpu_mem, mem_size);
        if (new_gpu_block == NULL) {
            outkey = LRU_out(LRU, GPU_id);
            if (outkey == NULL) {
                traverse_LRU_se(LRU);
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                //printf("free mem:%lu, total_mem:%lu\n", free_mem/1000000, total_mem/1000000);
                blasx_gpu_malloc_fini(LRU->gpu_mem, GPU_id);
                //printf("NOT ENOUGH MEM BUFFER!!!!!!!\n");
                exit(1);
            }
        }
#else
//CUDA MALLOC
        cuda_err = cudaSetDevice(GPU_id);
        assert(cuda_err == cudaSuccess);
        cuda_err = cudaMalloc(&new_gpu_block, mem_size);
        if (cudaErrorMemoryAllocation == cuda_err) {
            outkey = LRU_out(LRU, GPU_id);
            if (outkey == NULL) {
                traverse_LRU_se(LRU);
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                //printf("free mem:%lu, total_mem:%lu\n", free_mem/1000000, total_mem/1000000);
            }
            assert(outkey != NULL);
        }
#endif
        //printf("new_gpu_block:%p\n", new_gpu_block);
    }
    
    new_head->GPU_p     = new_gpu_block;
    rbt_insert(key, new_head, &(LRU->hash_map));
    if (LRU->head == NULL && LRU->tail == NULL) { //the first one
        //printf(" new head\n");
        LRU->head = new_head;
        LRU->tail = new_head;
    } else {
        //printf(" insert@front\n");
        LRU_elem* old_head = LRU->head;           //push front
        old_head->prior    = new_head;
        new_head->next     = old_head;
        LRU->head          = new_head;
    }
    return new_gpu_block;
}

int LRU_find(T key, LRU_t *LRU) {
    if (rbt_find(key, &(LRU->hash_map)) == NULL) {
        return 1; // found
    } else {
        return 0; // not found
    }
}

T LRU_out(LRU_t* LRU, int GPU_id) {
    if (LRU->tail == NULL) return NULL;
    if (LRU->tail->read_tracker == 0) {
        //behave as LRU, delete from the end
        //fprintf(stderr, "=>OUT=>%p type 1 reader:%d done:%d\n", LRU->tail->key, LRU->tail->read_tracker, LRU->tail->is_trans_done);
        if (LRU->tail != NULL && LRU->tail->prior != NULL) {
            LRU_elem *old_tail  = LRU->tail;
            LRU_elem *new_tail  = old_tail->prior;
            new_tail->next      = NULL;
            LRU->tail           = new_tail;
            void *key           = old_tail->key;
            //delete the LRU block from rbt
            rbt_node* to_delete = rbt_find(key, &LRU->hash_map);
            rbt_delete(to_delete, &LRU->hash_map);
            //free the gpu memory
#ifdef BLASX_MALLOC
            blasx_gpu_free(LRU->gpu_mem, old_tail->GPU_p);
#else
            cuda_err = cudaFree(old_tail->GPU_p);
            assert(cuda_err == cudaSuccess);
#endif
            old_tail->key   = NULL;
            old_tail->GPU_p = NULL;
            old_tail->prior = NULL;
            old_tail->next  = NULL;
            free(old_tail);
            return key;
        } else if(LRU-> tail != NULL && LRU->tail->prior == NULL) {
            LRU_elem *old_tail  = LRU->tail;
            T key               = old_tail->key;
            LRU->head           = NULL;
            LRU->tail           = NULL;
            rbt_node* to_delete = rbt_find(key, &LRU->hash_map);
            rbt_delete(to_delete, &LRU->hash_map); // delete from RBT
            //free the gpu memory
#ifdef BLASX_MALLOC
            blasx_gpu_free(LRU->gpu_mem, old_tail->GPU_p);
#else
            cuda_err = cudaFree(old_tail->GPU_p);
            assert(cuda_err == cudaSuccess);
#endif
            //fprintf(stderr, "=>OUT=>%p type 2 rbt_delete: %p rbt_key: %p\n", key, to_delete->associated_LRU_elem, to_delete->key);
            old_tail->key   = NULL;
            old_tail->GPU_p = NULL;
            old_tail->prior = NULL;
            old_tail->next  = NULL;
            free(old_tail);
            return key;
        } else {
            return NULL;
        }
    } else {
        // find the first zero reader node from the end
        LRU_elem* target = LRU->tail;
        //fprintf(stderr, "=>OUT=> type 3\n");
        while (target->read_tracker != 0) {
            target = target->prior;
            if (target == NULL) return NULL;
//            assert(target != NULL);
        }
        //printf("find the zero reader element\n");
        T key = target->key;
        //fprintf(stderr, "=>OUT=>%p type 1 reader:%d done:%d\n", target->key, target->read_tracker, target->is_trans_done);
        //fprintf(stderr, "=>OUT=>%p type 3\n", key);
        if (target == LRU->head) { //need update the head
            LRU_elem* old_head = target;
            LRU->head          = old_head->next;
            LRU->head->prior   = NULL;
            rbt_node* to_delete = rbt_find(key, &LRU->hash_map);
            rbt_delete(to_delete, &LRU->hash_map);            // delete from RBT
#ifdef BLASX_MALLOC
            blasx_gpu_free(LRU->gpu_mem, old_head->GPU_p);  // delete from gpumem
#else
            cuda_err = cudaFree(old_head->GPU_p);
            assert(cuda_err == cudaSuccess);
#endif
            old_head->key   = NULL;
            old_head->GPU_p = NULL;
            old_head->prior = NULL;
            old_head->next  = NULL;
            free(old_head);
        } else {                   //in the middle of LRU
            LRU_elem* to_free    = target;
            to_free->prior->next = to_free->next;
            to_free->next->prior = to_free->prior;
            rbt_node* to_delete = rbt_find(key, &LRU->hash_map);
            rbt_delete(to_delete, &LRU->hash_map);          // delete from RBT
#ifdef BLASX_MALLOC
            blasx_gpu_free(LRU->gpu_mem, to_free->GPU_p);   // delete from gpumem
#else
            cuda_err = cudaFree(to_free->GPU_p);
            assert(cuda_err == cudaSuccess);
#endif
            to_free->key   = NULL;
            to_free->GPU_p = NULL;
            to_free->prior = NULL;
            to_free->next  = NULL;
            free(to_free);
        }
        return key;
    }
}

//int main() {
//    LRU_t *LRU = LRU_init(0);
//    int i = 0;
//    double data[4000];
//    for (i = 0; i < 30; i++) {
//        printf("-----------------------\n");
//        int index = rand()%30;
//        printf("index: %d ", index);
//        if (index < 20) {
//            LRU_out(LRU, 0);
//        }
//        double* key = &data[index];
//        void* gpu_pointer = LRU_in(key, LRU, sizeof(float)*1024*1024, 0);
//        if (index < 15) {
//             (rbt_find(key, &(LRU->hash_map))->associated_LRU_elem->read_tracker)++;
//        }
//        printf("gpu_pointer %p\n", gpu_pointer);
//        traverse_LRU_se(LRU);
//        traverse_LRU_es(LRU);
//        printf("-----------------------\n\n");
//    }
//    LRU_free(LRU, 0);
//    return 0;
//}



