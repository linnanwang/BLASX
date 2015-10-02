/* red-black tree */
#include <red_black_tree.h>
#define NIL &sentinel             /* all leafs are sentinels */
rbt_node sentinel = { NIL, NIL, 0, NULL, BLACK, NULL};

void rotateLeft(rbt_node *x, rbt_node **root) {

   /**************************
    *  rotate node x to left *
    **************************/

    rbt_node *y = x->right;

    /* establish x->right link */
    x->right = y->left;
    if (y->left != NIL) y->left->parent = x;

    /* establish y->parent link */
    if (y != NIL) y->parent = x->parent;
    if (x->parent) {
        if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
    } else {
        *root = y;
    }

    /* link x and y */
    y->left = x;
    if (x != NIL) x->parent = y;
}

void rotateRight(rbt_node *x, rbt_node **root) {

   /****************************
    *  rotate node x to right  *
    ****************************/

    rbt_node *y = x->left;

    /* establish x->left link */
    x->left = y->right;
    if (y->right != NIL) y->right->parent = x;

    /* establish y->parent link */
    if (y != NIL) y->parent = x->parent;
    if (x->parent) {
        if (x == x->parent->right)
            x->parent->right = y;
        else
            x->parent->left = y;
    } else {
        *root = y;
    }

    /* link x and y */
    y->right = x;
    if (x != NIL) x->parent = y;
}

void insertFixup(rbt_node *x, rbt_node **root) {

   /*************************************
    *  maintain Red-Black tree balance  *
    *  after inserting node x           *
    *************************************/

    /* check Red-Black properties */
    while (x != *root && x->parent->color == RED) {
        /* we have a violation */
        if (x->parent == x->parent->parent->left) {
            rbt_node *y = x->parent->parent->right;
            if (y->color == RED) {

                /* uncle is RED */
                x->parent->color = BLACK;
                y->color = BLACK;
                x->parent->parent->color = RED;
                x = x->parent->parent;
            } else {

                /* uncle is BLACK */
                if (x == x->parent->right) {
                    /* make x a left child */
                    x = x->parent;
                    rotateLeft(x, root);
                }

                /* recolor and rotate */
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                rotateRight(x->parent->parent, root);
            }
        } else {

            /* mirror image of above code */
            rbt_node *y = x->parent->parent->left;
            if (y->color == RED) {

                /* uncle is RED */
                x->parent->color = BLACK;
                y->color = BLACK;
                x->parent->parent->color = RED;
                x = x->parent->parent;
            } else {

                /* uncle is BLACK */
                if (x == x->parent->left) {
                    x = x->parent;
                    rotateRight(x, root);
                }
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                rotateLeft(x->parent->parent, root);
            }
        }
    }
    (*root)->color = BLACK;
}

rbt_node* rbt_insert(T key, LRU_elem *associated_LRU_elem, rbt_node **root) {
    rbt_node *current, *parent, *x;

   /***********************************************
    *  allocate node for data and insert in tree  *
    ***********************************************/

    /* find where node belongs */
    current = *root;
    parent = 0;
    while (current != NIL) {
        if (compEQ(key, current->key)) return (current);
        parent = current;
        current = compLT(key, current->key) ?
            current->left : current->right;
    }

    /* setup new node */
    if ((x = malloc (sizeof(*x))) == 0) {
        printf ("insufficient memory (insertNode)\n");
        exit(1);
    }
    x->key    = key;
    x->parent = parent;
    x->left   = NIL;
    x->right  = NIL;
    x->color  = RED;
    x->associated_LRU_elem = associated_LRU_elem;

    /* insert node in tree */
    if(parent) {
        if(compLT(key, parent->key))
            parent->left = x;
        else
            parent->right = x;
    } else {
        *root = x;
    }

    insertFixup(x, root);
    return(x);
}

void deleteFixup(rbt_node *x, rbt_node **root) {

   /*************************************
    *  maintain Red-Black tree balance  *
    *  after deleting node x            *
    *************************************/

    while (x != *root && x->color == BLACK) {
        if (x == x->parent->left) {
            rbt_node *w = x->parent->right;
            if (w->color == RED) {
                w->color = BLACK;
                x->parent->color = RED;
                rotateLeft (x->parent, root);
                w = x->parent->right;
            }
            if (w->left->color == BLACK && w->right->color == BLACK) {
                w->color = RED;
                x = x->parent;
            } else {
                if (w->right->color == BLACK) {
                    w->left->color = BLACK;
                    w->color = RED;
                    rotateRight (w, root);
                    w = x->parent->right;
                }
                w->color = x->parent->color;
                x->parent->color = BLACK;
                w->right->color = BLACK;
                rotateLeft (x->parent, root);
                x = *root;
            }
        } else {
            rbt_node *w = x->parent->left;
            if (w->color == RED) {
                w->color = BLACK;
                x->parent->color = RED;
                rotateRight (x->parent, root);
                w = x->parent->left;
            }
            if (w->right->color == BLACK && w->left->color == BLACK) {
                w->color = RED;
                x = x->parent;
            } else {
                if (w->left->color == BLACK) {
                    w->right->color = BLACK;
                    w->color = RED;
                    rotateLeft (w, root);
                    w = x->parent->left;
                }
                w->color = x->parent->color;
                x->parent->color = BLACK;
                w->left->color = BLACK;
                rotateRight (x->parent, root);
                x = *root;
            }
        }
    }
    x->color = BLACK;
}

void rbt_delete(rbt_node *z, rbt_node **root) {
    rbt_node *x, *y;

   /*****************************
    *  delete node z from tree  *
    *****************************/

    if (!z || z == NIL) return;


    if (z->left == NIL || z->right == NIL) {
        /* y has a NIL node as a child */
        y = z;
    } else {
        /* find tree successor with a NIL node as a child */
        y = z->right;
        while (y->left != NIL) y = y->left;
    }

    /* x is y's only child */
    if (y->left != NIL)
        x = y->left;
    else
        x = y->right;

    /* remove y from the parent chain */
    x->parent = y->parent;
    if (y->parent)
        if (y == y->parent->left)
            y->parent->left = x;
        else
            y->parent->right = x;
    else
    {
        *root = x;
    }

    if (y != z) {
        z->key = y->key;
        //CHANGE
        z->associated_LRU_elem = y->associated_LRU_elem;
    }


    if (y->color == BLACK)
        deleteFixup (x, root);
    
    y->key = NULL;
    y->associated_LRU_elem = NULL;
    y->left = NULL;
    y->right = NULL;
    y->parent = NULL;
    free (y);
}

rbt_node* rbt_find(T key, rbt_node **root) {

   /*******************************
    *  find node containing data  *
    *******************************/
    rbt_node *current = *root;
    while(current != NIL || current == NULL)
        if(key == current->key){
            return (current);
        }
        else
            current = compLT (key, current->key) ?
                current->left : current->right;
    return NULL;
}

void traverse_auxiliary_kernel(rbt_node* n) {  // in order tree traverse
    if( n != NIL) {
        //printf("@node:%d going left %p\n", n->key, n->left);
        traverse_auxiliary_kernel(n->left);
        printf("key:%p LRU_ele:%p left:%p rigth:%p parent:%p\n", n->key, n->associated_LRU_elem, n->left, n->right, n->parent);
        //printf("@node:%d going right %p\n", n->key, n->right);
        traverse_auxiliary_kernel(n->right);
    }
}

void rbt_free_kernel(rbt_node** n) {  // in order tree traverse
    if( *n != NIL ) {
        rbt_free(&(*n)->left);
        rbt_free(&(*n)->right);
//        printf("%p key:%p left%p right%p parent:%p\n", *n, (*n)->key, (*n)->left, (*n)->right, (*n)->parent);
        (*n)->left   = NULL;
        (*n)->right  = NULL;
        (*n)->parent = NULL;
        (*n)->associated_LRU_elem = NULL;
        (*n)->key    = NULL;
        free(*n);
    }
}

void rbt_free(rbt_node** root) {
    rbt_free_kernel(root);
    *root = NIL;
}


void rbt_traverse(rbt_node *root) {
    printf("=====RBT traverse=====\n");
    traverse_auxiliary_kernel(root);
    printf("======================\n");
}

int find_max_depth_kernel(rbt_node* current) {
    int current_depth;
    if(current == NIL) {
        return 0;
    } else {
        current_depth = 1 + MAX(find_max_depth_kernel(current->left), find_max_depth_kernel(current->right));
    }
    return current_depth;
}

void find_max_depth(rbt_node *root) {
    int max_depth = find_max_depth_kernel(root);
    printf("the max depth %d\n", max_depth);
}

rbt_node* rbt_init() {
    rbt_node* root = NIL;
    return root;
}

//int main(int argc, char **argv) {
//    int a, maxnum, ct;
//    int trial = 0;
//    for (trial = 0; trial < 3000; trial++) {
//        fprintf(stderr, "trial:%d ", trial);
//        rbt_node *root =  rbt_init();
//        rbt_node *t;
//        double data[4000];
//        for (ct = 0; ct < 4000; ct++) {
//            int index = ct%200;
//            double *curt = &data[index];
//            rbt_node* is_exist = rbt_find(curt, &root);
//            if (is_exist == NULL) {
//                rbt_insert(curt, NULL, &root);
//            }
//        }
////        traverse(root);
//        find_max_depth(root);
//        rbt_free(&root);
//    }
//}
