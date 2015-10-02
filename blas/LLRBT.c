#include <stdlib.h>
#include <stdio.h>
#define MAX(a,b)   ((a < b) ?  (b) : (a))

//--------auxiliary operation--------//

enum LLRBTNodeColor {RED, BLACK};

typedef struct LLRBT_node {
    struct LLRBT_node* left;
    struct LLRBT_node* right;
    void *key;
    enum LLRBTNodeColor incoming_link_color;  // color 0:black, 1:red
} node;

void LLRBT_destruction_auxiliary(node* n) {  // post-order is suitable for destruction
    if( n != NULL ) {
        LLRBT_destruction_auxiliary(n->left);
        LLRBT_destruction_auxiliary(n->right);
        free(n);
    }
}

node* LLRBT_right_rotation(node* head) {   // CATION: this is a in place algorithm
    if ( head->left == NULL ) {
        printf("invalid right rotation since no node on the right");
        return head;
    } else {
        node* new_head = head->left;
        head->left         = new_head->right;
        new_head->right    = head;
        return new_head;
    }
}

node* LLRBT_left_rotation(node* head) {
    if(head->right != NULL) {
        node* new_head = head->right;
        head->right        = new_head->left;
        new_head->left     = head;
        return new_head;
    } else {
        printf("no more right node for rotation!@%p\n", head->key);
        return head;
    }
}
    
node* split_4_nodes(node* current) {
    current = LLRBT_right_rotation(current);
    current->left->incoming_link_color = BLACK;
    return current;
}

node* insert_kernel(node* current,  void *key) {
    if ( current == NULL ) {
        node* new_node = (node*) malloc(sizeof(node));
        new_node->left                =  NULL;
        new_node->right               =  NULL;
        new_node->incoming_link_color =  RED;
        new_node->key                 =  key;
        return new_node;
    }
    //check split
    if( current->left != NULL ){
        if( current->left->incoming_link_color == RED ) {
            if( current->left->left != NULL ) {
                if( current->left->left->incoming_link_color == RED ) {
                    current = split_4_nodes(current);
                }
            }
        }
    }
    
    if( key >= current->key ) current->right = insert_kernel(current->right, key);
    else {
        current->left = insert_kernel(current->left, key);
    }
    
    if( current->right != NULL && current->right->incoming_link_color == RED ) {  // maintain the left-lean property
        current                            = LLRBT_left_rotation(current);
        current->incoming_link_color       = current->left->incoming_link_color;
        current->left->incoming_link_color = RED;
    }
    
    return current;
}

void traverse_auxiliary_kernel(node* n) {  // in order tree traverse
    if( n != NULL ) {
        //printf("@node:%d going left %p\n", n->key, n->left);
        traverse_auxiliary_kernel(n->left);
        printf("node:%p, addr %p color:%d\n", n->key, n, n->incoming_link_color);
        //printf("@node:%d going right %p\n", n->key, n->right);
        traverse_auxiliary_kernel(n->right);
    }
}

int find_max_depth_kernel(node* current) {
    int current_depth;
    if( current==NULL ) {
        return 0;
    } else {
        current_depth = 1 + MAX(find_max_depth_kernel(current->left), find_max_depth_kernel(current->right));
    }
    return current_depth;
}

node* find_kernel(node* curt, void const *key) {
    if(curt == NULL) return NULL;
    if (curt->key == key) {
        return curt;
    } else if(curt->key < key) {
        return find_kernel(curt->right, key);
    } else {
        return find_kernel(curt->left,  key);
    }
}
//------------------------------------//

node* RBT_init() {
    node* root = NULL;
    return root;
}

void LLRBT_destruct(node *root) {
    LLRBT_destruction_auxiliary(root);
}

node* insert(void *key, node *root) {
    root = insert_kernel(root, key);
    root->incoming_link_color = BLACK;
    return root;
}

node* find(void const *key, node *root) {
    return find_kernel(root, key);
}

void traverse(node *root) {
    traverse_auxiliary_kernel(root);
}

void find_max_depth(node *root) {
    int max_depth = find_max_depth_kernel(root);
    printf("the max depth of this tree is:%d\n", max_depth);
}

int main() {
    node* t = RBT_init();
    float data[20000];
    int i = 0;
    for(i = 19999; i >= 0; i--) {
        void *data_addr = (void*) &data[i];
        t = insert(data_addr, t);
    }
//    traverse(t);
    printf("target key%p\n", &data[7]);
    node *target = find((void*)&data[7], t);
    printf("target key%p\n", target->key);
    find_max_depth(t);
}

