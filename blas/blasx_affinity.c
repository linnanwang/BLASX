#include <blasx_affinity.h>

int blasx_set_affinity(int GPU_id) {
#ifdef affinity
    int err;
    int threadId = GPU_id + 1;
    cpu_set_t cpuset;
    pthread_t self;
    self = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(threadId, &cpuset);
    err = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
    if (err != 0) printf("pthread_setaffinity_np");
    err = pthread_getaffinity_np(self, sizeof(cpu_set_t), &cpuset);
    if (err != 0) printf("pthread_getaffinity_np");
    return err;
#else
    return 0;
#endif
}