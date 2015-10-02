#ifndef BLASX_MUTEX
#define BLASX_MUTEX
#if defined(_WIN64)
/* Microsoft Windows (64-bit). ------------------------------ */

#elif defined(_WIN32)
/* Microsoft Windows (32-bit). ------------------------------ */
//POSIX machine
#endif
#include <stdio.h>
#include <pthread.h>

typedef struct{
    pthread_mutex_t lock;
}blasx_mutex_t;
extern blasx_mutex_t mutex;

int blasx_mutex_init();

int blasx_mutex_destroy();

int blasx_mutex_lock();

int blasx_mutex_unlock();

#endif /* BLASX_MUTEX */

