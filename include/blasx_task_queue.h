#ifndef STANDARD_LIB_H
#define STANDARD_LIB_H
#include <stdio.h>
#include <stdlib.h>
#endif

#ifndef TASK_QUEUE_H
#define TASK_QUEUE_H
typedef struct queue_t {
    int TAIL;
}queue_t;
void init_queue(queue_t* q, int task_num);
int dequeue(volatile queue_t* q);
#endif /* TASK_QUEUE_H */
