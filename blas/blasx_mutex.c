#include <blasx_mutex.h>

blasx_mutex_t mutex = { PTHREAD_MUTEX_INITIALIZER };

int blasx_mutex_init(blasx_mutex_t *mutex)
{
    if (pthread_mutex_init(&mutex->lock, NULL) != 0)
    {
        fprintf(stderr,"ERROR: fail on mutex construction!\n");
        return 1;
    }
    return 0;
}

int blasx_mutex_destroy()
{
    if (pthread_mutex_destroy(&mutex.lock) != 0) {
        fprintf(stderr,"ERROR: fail on mutex destruction!\n");
        return 1;
    }else{
        Blasx_Debug_Output("Posix Mutex Destroied!\n");
    }
    return 0;
}

int blasx_mutex_lock()
{
    
    pthread_mutex_lock(&mutex.lock);
    return 0;
}

int blasx_mutex_unlock()
{
    pthread_mutex_unlock(&mutex.lock);
    return 0;
}
