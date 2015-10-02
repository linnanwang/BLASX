#ifndef RED_BLACK_TREE_H
#define RED_BLACK_TREE_H
#include <RBT_LRU_common.h>

/* Red-Black Tree description */
typedef enum { BLACK, RED } rbt_node_color;
struct node_ {
    struct node_ *left;
    struct node_ *right;
    struct node_ *parent;
    LRU_elem *associated_LRU_elem;
    rbt_node_color color;
    T key;
};
rbt_node* rbt_init();
rbt_node* rbt_find(T key, rbt_node **root);
rbt_node* rbt_insert(T key, LRU_elem *associated_LRU_elem, rbt_node **root);
void rbt_free(rbt_node** root);
void rbt_delete(rbt_node *z, rbt_node **root);
void find_max_depth(rbt_node *root);
void rbt_traverse(rbt_node *root);

#endif /* RED_BLACK_TREE_H */