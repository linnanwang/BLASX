#ifndef AFFINITY_H
#define AFFINITY_H
#define _GNU_SOURCE

#ifdef affinity
#include <sched.h>
#endif
int blasx_set_affinity(int GPU_id);

#endif /* AFFINITY_H */