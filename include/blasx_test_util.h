#ifndef STANDARD_LIB_H
#define STANDARD_LIB_H
#include <stdio.h>
#include <stdlib.h>
#endif

#ifndef TEST_UTIL_H
#define TEST_UTIL_H
#include <math.h> 
#include <assert.h>
#include <stdarg.h>
#include <sys/time.h>

double get_cur_time();
void output(const char *fmt, ...);
//double util
void rand_fill(double* mat, int nrow, int ncol);
void print( double* mat, int nrow, int ncol, char* c);
void print_mat( double* mat, int nrow, int ncol, char* c);
void fill_trsm(double* mat, int nrow, int ncol, double target);
void deepcopy_matrix( double *mat_source, double *mat_destination, int nrow, int ncol);
void verify_matrix_calculation(double *mat1, double *mat2, int nrow, int ncol, char *c);
void block_verify(double *correct_mat, int ld, int i, int j, double *block, int block_dim);
//single util
void deepcopy_float( float *mat_source, float *mat_destination, int nrow, int ncol);
void verify_matrix_calculation_float(float *mat1, float *mat2, int nrow, int ncol, char *c);
void rand_fill_float(float* mat, int nrow, int ncol);

#endif /* TEST_UTIL_H */