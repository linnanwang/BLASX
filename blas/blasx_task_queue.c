#include <blasx_task_queue.h>

void init_queue(queue_t* q, int task_num) {
    q->TAIL  = task_num;
}

int dequeue(volatile queue_t* q) {
    while (1) {
        if (q->TAIL < 0) {
            return -1;
        }
        int old_tail = q->TAIL;
        if ( __sync_bool_compare_and_swap(&q->TAIL, old_tail, old_tail-1) ) {
            return old_tail;
        }
    }
}