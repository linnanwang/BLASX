#ifndef STANDARD_LIB_H
#define STANDARD_LIB_H
#include <stdio.h>
#include <stdlib.h>
#endif

#ifndef RBT_LRU_COMMON_H
#define RBT_LRU_COMMON_H
/* common functions*/
#define MAX(a,b)   ((a < b) ?  (b) : (a))
typedef void* T;                  /* type of item to be stored */
typedef enum { SINGLE, DOUBLE, SINGLE_COMPLEX, DOUBLE_COMPLEX } pointer_t;
#define compLT(a,b) (a < b)
#define compEQ(a,b) (a == b)

//forward declare
typedef struct node_ rbt_node;
typedef struct LRU_element LRU_elem;
typedef struct LRU_type    LRU_t;

#endif /* RBT_LRU_COMMON */

