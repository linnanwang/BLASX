#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <f77blas.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>

void Fill_Double(double *mat, int size1, int size2)
{
    int i;
    for (i = 0; i< (double)size1*(double)size2; i++) {
        mat[i] = (double)rand()/(double)RAND_MAX;
    }
}

void Fill_Float(float *mat, int size1, int size2)
{
    int i;
    for (i = 0; i< (double)size1*(double)size2; i++) {
        mat[i] = (float)rand()/(float)RAND_MAX;
    }
}

int
main(int argc, char **argv)
{

    int loop = 0;
    for (loop = 1; loop < 4; loop++) {
        int M = 6384;
        int N = M;
        int K = M;
        float alpha_f = (float)(((double) rand()/(double)RAND_MAX)*10)+1;
        float beta_f  = (float)(((double) rand()/(double)RAND_MAX)*10)+1;
        float *A_f, *B_f, *C_f;
        A_f = (float*)malloc(sizeof(float)*M*K);
        B_f = (float*)malloc(sizeof(float)*K*N);
        C_f = (float*)malloc(sizeof(float)*M*N);
        Fill_Float(A_f,M,K);
        Fill_Float(B_f,K,N);
        Fill_Float(C_f,M,N);
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,N,K,
                    alpha_f,A_f,M,
                    B_f,K,
                    beta_f,C_f,M);
        free(A_f);
        free(B_f);
        free(C_f);
    }
    
    for (loop = 1; loop < 4; loop++) {
        int M = 6384;
        int N = M;
        int K = M;
        double alpha_f = (double)(((double) rand()/(double)RAND_MAX)*10)+1;
        double beta_f  = (double)(((double) rand()/(double)RAND_MAX)*10)+1;
        double *A_f, *B_f, *C_f;
        A_f = (double*)malloc(sizeof(double)*M*K);
        B_f = (double*)malloc(sizeof(double)*K*N);
        C_f = (double*)malloc(sizeof(double)*M*N);
        Fill_Double(A_f,M,K);
        Fill_Double(B_f,K,N);
        Fill_Double(C_f,M,N);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,N,K,
                    alpha_f,A_f,M,
                    B_f,K,
                    beta_f,C_f,M);
        free(A_f);
        free(B_f);
        free(C_f);
    }

}
