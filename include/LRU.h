#ifndef LRU_H
#define LRU_H
#include <red_black_tree.h>
#include <cuda_runtime.h>
#include <gpu_malloc.h>
#include <assert.h>

/* LRU description */
struct LRU_element {
    T key; //CPU_p
    void *GPU_p;
    int    read_tracker;
    struct LRU_element *prior;
    struct LRU_element *next;
    int    is_trans_done;
};

struct LRU_type {
    rbt_node *hash_map;
    blasx_gpu_malloc_t *gpu_mem;
    LRU_elem *head;
    LRU_elem *tail;
};

LRU_t* LRU_init(int GPU_id);
void LRU_free(LRU_t *LRU, int GPU_id);
T    LRU_in(T key, LRU_t *LRU, size_t mem_size, int GPU_id);
T    LRU_out(LRU_t *LRU, int GPU_id);
T    LRU_reorder(T key, LRU_t* LRU);
int  LRU_find(T key, LRU_t *LRU);
void traverse_LRU_se(const LRU_t *LRU );
void traverse_LRU_es(const LRU_t *LRU );

#endif /* LRU_H */