#ifndef STANDARD_LIB_H
#define STANDARD_LIB_H
#include <stdio.h>
#include <stdlib.h>
#endif

#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H
#include <blasx_common.h>

#define BLASX_GPU_MEM_MAX_SEGMENT    10000
#define BLASX_GPU_INIT_MEM 1000000*50

/*---gpu_malloc---*/
typedef struct segment {
    void  *addr;       /* Address of the first byte of this segment */
    size_t mem_size;   /* Size of memory occupied by this segment */
    size_t mem_free;   /* Size of memory free after this segment */
    struct segment *next;
    struct segment *prev;
} blasx_gpu_segment_t;

typedef struct gpu_malloc_s {
    void                *base;                 /* Base pointer              */
    blasx_gpu_segment_t *allocated_segments;   /* List of allocated segment */
    blasx_gpu_segment_t *free_segments;        /* List of available segment */
    size_t               total_size;           /* total memory size ocupied */
    int                  max_segment;          /* Maximum number of segment */
} blasx_gpu_malloc_t;

blasx_gpu_malloc_t* blasx_gpu_malloc_init();
void   blasx_gpu_malloc_fini(blasx_gpu_malloc_t* gdata, int GPU_id);
void*  blasx_gpu_malloc(blasx_gpu_malloc_t *gdata, size_t nbytes);
void   blasx_gpu_free(blasx_gpu_malloc_t *gdata, void *addr);

#endif /* GPU_MALLOC_H */
