#ifndef BLASX_GLOBALPOINTERS
#define BLASX_GLOBALPOINTERS
#include <cblas.h>

#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef __cplusplus
extern "C" {
#endif
#define blasxDoubleComplex		void
#define blasxDoubleComplexF77 	double
#define blasxFloatComplex 		void
#define blasxFloatComplexF77	float

/****************Configuration*********************/
typedef struct { float real, imag; } blasx_complex_float;
typedef struct { double real, imag; } blasx_complex_double;
/*---------mutex--------*/
/*********************LEVEL1***********************/
//AMAX
extern size_t (*cblas_isamax_p)(const int N, const float  *X, const int incX);
extern size_t (*cblas_idamax_p)(const int N, const double *X, const int incX);
extern size_t (*cblas_icamax_p)(const int N, const void   *X, const int incX);
extern size_t (*cblas_izamax_p)(const int N, const void   *X, const int incX);
    
//SWAP
extern void (*cblas_sswap_p)(const int N, float *X, const int incX,
                             float *Y, const int incY);
extern void (*cblas_dswap_p)(const int N, double *X, const int incX,
                             double *Y, const int incY);
extern void (*cblas_cswap_p)(const int N, void *X, const int incX,
                             void *Y, const int incY);
extern void (*cblas_zswap_p)(const int N, void *X, const int incX,
                             void *Y, const int incY);

//SCAL
extern void (*cblas_sscal_p)(const int N, const float alpha, float *X, const int incX);
extern void (*cblas_dscal_p)(const int N, const double alpha, double *X, const int incX);
extern void (*cblas_cscal_p)(const int N, const void *alpha, void *X, const int incX);
extern void (*cblas_zscal_p)(const int N, const void *alpha, void *X, const int incX);
extern void (*cblas_csscal_p)(const int N, const float alpha, void *X, const int incX);
extern void (*cblas_zdscal_p)(const int N, const double alpha, void *X, const int incX);

//ROTMG
extern void (*cblas_drotmg_p)(double *d1, double *d2, double *b1, const double b2, double *P);
extern void (*cblas_srotmg_p)(float *d1, float *d2, float *b1, const float b2, float *P);

//ROTM
extern void (*cblas_srotm_p)(const int N, float *X, const int incX,
                             float *Y, const int incY, const float *P);
extern void (*cblas_drotm_p)(const int N, double *X, const int incX,
                             double *Y, const int incY, const double *P);


//ROTG
extern void (*cblas_srotg_p)(float *a, float *b, float *c, float *s);
extern void (*cblas_drotg_p)(double *a, double *b, double *c, double *s);

//ROT
extern void (*cblas_srot_p)(const int N, float *X, const int incX,
                            float *Y, const int incY, const float c, const float s);
extern void (*cblas_drot_p)(const int N, double *X, const int incX,
                            double *Y, const int incY, const double c, const double  s);
extern void (*csrot_p)(int *n, float *X, int *incX, float *Y, int *incy, float *c, float *s);
extern void (*zdrot_p)(int *n, double *X, int *incX, double *Y, int *incy, double *c, double *s);


//NRM2
extern float (*cblas_snrm2_p)(const int N, const float *X, const int incX);
extern double (*cblas_dnrm2_p)(const int N, const double *X, const int incX);
extern float (*cblas_scnrm2_p)(const int N, const void *X, const int incX);
extern double (*cblas_dznrm2_p)(const int N, const void *X, const int incX);

    
//DOTU
extern void (*cblas_zdotu_sub_p)(const int N, const void *X, const int incX,
                                 const void *Y, const int incY, void *dotu);
extern void (*cblas_cdotu_sub_p)(const int N, const void *X, const int incX,
                                 const void *Y, const int incY, void *dotu);

//DOTC
extern void (*cblas_zdotc_sub_p)(const int N, const void *X, const int incX,
                                 const void *Y, const int incY, void *dotc);
extern void (*cblas_cdotc_sub_p)(const int N, const void *X, const int incX,
                                 const void *Y, const int incY, void *dotc);
//SDOT
extern float (*cblas_sdsdot_p)(const int N, const float alpha, const float *X,
                               const int incX, const float *Y, const int incY);
extern double (*cblas_dsdot_p)(const int N, const float *X, const int incX, const float *Y,
                               const int incY);

//DOT
extern float (*cblas_sdot_p)(const int N, const float  *X, const int incX,
                             const float  *Y, const int incY);
extern double (*cblas_ddot_p)(const int N, const double *X, const int incX,
                              const double *Y, const int incY);
    
//COPY
extern void (*cblas_scopy_p)(const int N, const float *X, const int incX,
                             float *Y, const int incY);
extern void (*cblas_dcopy_p)(const int N, const double *X, const int incX,
                             double *Y, const int incY);
extern void (*cblas_ccopy_p)(const int N, const void *X, const int incX,
                             void *Y, const int incY);
extern void (*cblas_zcopy_p)(const int N, const void *X, const int incX,
                             void *Y, const int incY);

//AXPY
extern void (*cblas_saxpy_p)(const int N, const float alpha, const float *X,
                             const int incX, float *Y, const int incY);
extern void (*cblas_daxpy_p)(const int N, const double alpha, const double *X,
                             const int incX, double *Y, const int incY);
extern void (*cblas_caxpy_p)(const int N, const void *alpha, const void *X,
                             const int incX, void *Y, const int incY);
extern void (*cblas_zaxpy_p)(const int N, const void *alpha, const void *X,
                             const int incX, void *Y, const int incY);



//ASUM
extern float (*cblas_sasum_p)(const int N, const float *X, const int incX);
extern double (*cblas_dasum_p)(const int N, const double *X, const int incX);
extern double (*cblas_dzasum_p)(const int N, const void *X, const int incX);
extern float (*cblas_scasum_p)(const int N, const void *X, const int incX);


/*********************LEVEL2***********************/
//SYR2
extern void (*cblas_dsyr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const double alpha, const double *X,
                             const int incX, const double *Y, const int incY, double *A,
                             const int lda);

extern void (*cblas_ssyr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const float alpha, const float *X,
                             const int incX, const float *Y, const int incY, float *A,
                             const int lda);


//SYR
extern void (*cblas_dsyr_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const int N, const double alpha, const double *X,
                            const int incX, double *A, const int lda);
    
extern void (*cblas_ssyr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                            const int N, const float alpha, const float *X,
                            const int incX, float *A, const int lda);

//SYMV
extern void (*cblas_dsymv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const double alpha, const double *A,
                             const int lda, const double *X, const int incX,
                             const double beta, double *Y, const int incY);
    
extern void (*cblas_ssymv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const float alpha, const float *A,
                             const int lda, const float *X, const int incX,
                             const float beta, float *Y, const int incY);

//SPR2
extern void (*cblas_dspr2_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                             const int N, const double alpha, const double *X,
                             const int incX, const double *Y, const int incY, double *A);

extern void (*cblas_sspr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const float alpha, const float *X,
                             const int incX, const float *Y, const int incY, float *A);
    
//SPR
extern void (*cblas_dspr_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const int N, const double alpha, const double *X,
                            const int incX, double *Ap);

extern void (*cblas_sspr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                            const int N, const float alpha, const float *X,
                            const int incX, float *Ap);
//SPMV
extern void (*cblas_dspmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const double alpha, const double *Ap,
                             const double *X, const int incX,
                             const double beta, double *Y, const int incY);

extern void (*cblas_sspmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const float alpha, const float *Ap,
                             const float *X, const int incX,
                             const float beta, float *Y, const int incY);


//SBMV
extern void (*cblas_dsbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const int K, const double alpha, const double *A,
                             const int lda, const double *X, const int incX,
                             const double beta, double *Y, const int incY);

extern void (*cblas_ssbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const int K, const float alpha, const float *A,
                             const int lda, const float *X, const int incX,
                             const float beta, float *Y, const int incY);

//GER
extern void (*cblas_dger_p)(const enum CBLAS_ORDER order, const int M, const int N,
                            const double alpha, const double *X, const int incX,
                            const double *Y, const int incY, double *A, const int lda);

extern void (*cblas_sger_p)(const enum CBLAS_ORDER order, const int M, const int N,
                            const float alpha, const float *X, const int incX,
                            const float *Y, const int incY, float *A, const int lda);
    
//TRSV
extern void (*cblas_ztrsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *A, const int lda, void *X,
                             const int incX);
    
extern void (*cblas_ctrsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *A, const int lda, void *X,
                             const int incX);
    
extern void (*cblas_dtrsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const double *A, const int lda, double *X,
                             const int incX);
    
extern void (*cblas_strsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const float *A, const int lda, float *X,
                             const int incX);


//TRMV
extern void (*cblas_ztrmv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *A, const int lda,
                             void *X, const int incX);
    
extern void (*cblas_ctrmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *A, const int lda,
                             void *X, const int incX);
    
extern void (*cblas_dtrmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const double *A, const int lda,
                             double *X, const int incX);
    
extern void (*cblas_strmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const float *A, const int lda,
                             float *X, const int incX);



//TPSV
extern void (*cblas_ztpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *Ap, void *X, const int incX);
    
extern void (*cblas_ctpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *Ap, void *X, const int incX);


extern void (*cblas_dtpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const double *Ap, double *X, const int incX);

extern void (*cblas_stpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const float *Ap, float *X, const int incX);


//TPMV
extern void (*cblas_ztpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *Ap, void *X, const int incX);
    
extern void (*cblas_ctpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const void *Ap, void *X, const int incX);

extern void (*cblas_dtpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const double *Ap, double *X, const int incX);
    
extern void (*cblas_stpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const float *Ap, float *X, const int incX);

//TBSV
extern void (*cblas_ztbsv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const void *A, const int lda,
                             void *X, const int incX);

extern void (*cblas_ctbsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const void *A, const int lda,
                             void *X, const int incX);
    
extern void (*cblas_dtbsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const double *A, const int lda,
                             double *X, const int incX);
    
extern void (*cblas_stbsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const float *A, const int lda,
                             float *X, const int incX);

//TBMV
extern void (*cblas_ztbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const void *A, const int lda,
                             void *X, const int incX);

extern void (*cblas_ctbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const void *A, const int lda,
                             void *X, const int incX);
    
extern void (*cblas_dtbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const double *A, const int lda,
                             double *X, const int incX);
    
extern void (*cblas_stbmv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                             const int N, const int K, const float *A, const int lda,
                             float *X, const int incX);

//HPR2
extern void (*cblas_zhpr2_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *Ap);
    
extern void (*cblas_chpr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *Ap);
    
//HPR
extern void (*cblas_zhpr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                            const int N, const double alpha, const void *X,
                            const int incX, void *A);
    
extern void (*cblas_chpr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                            const int N, const float alpha, const void *X,
                            const int incX, void *A);

//HPMV
extern void (*cblas_zhpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const void *alpha, const void *Ap,
                             const void *X, const int incX,
                             const void *beta, void *Y, const int incY);
    
extern void (*cblas_chpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const void *alpha, const void *Ap,
                             const void *X, const int incX,
                             const void *beta, void *Y, const int incY);

//HER2
extern void (*cblas_zher2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *A, const int lda);
    
extern void (*cblas_cher2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *A, const int lda);


//HER
extern void (*cblas_zher_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const int N, const double alpha, const void *X, const int incX,
                            void *A, const int ldA);

extern void (*cblas_cher_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                            const int N, const float alpha, const void *X, const int incX,
                            void *A, const int lda);

//HEMV
extern void (*cblas_zhemv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                             const int N, const void *alpha, const void *A,
                             const int ldA, const void *X, const int incX,
                             const void *beta, void *Y, const int incY);
    
extern void (*cblas_chemv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const void *alpha, const void *A,
                             const int lda, const void *X, const int incX,
                             const void *beta, void *Y, const int incY);

//HBMV
extern void (*cblas_zhbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const int K, const void *alpha, const void *A,
                             const int lda, const void *X, const int incX,
                             const void *beta, void *Y, const int incY);
    
extern void (*cblas_chbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const int K, const void *alpha, const void *A,
                             const int lda, const void *X, const int incX,
                             const void *beta, void *Y, const int incY);

//GERU
extern void (*cblas_zgeru_p)(const enum CBLAS_ORDER order, const int M, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *A, const int lda);
    
extern void (*cblas_cgeru_p)(const enum CBLAS_ORDER order, const int M, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *A, const int lda);

//GERC
extern void (*cblas_zgerc_p)(const enum CBLAS_ORDER Order, const int M, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *A, const int lda);

extern void (*cblas_cgerc_p)(const enum CBLAS_ORDER order, const int M, const int N,
                             const void *alpha, const void *X, const int incX,
                             const void *Y, const int incY, void *A, const int lda);

//GBMV
extern void (*cblas_zgbmv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const int KL, const int KU, const void *alpha,
                             const void *A, const int lda, const void *X,
                             const int incX, const void *beta, void *Y, const int incY);
    
extern void (*cblas_cgbmv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const int KL, const int KU, const void *alpha,
                             const void *A, const int lda, const void *X,
                             const int incX, const void *beta, void *Y, const int incY);
    
extern void (*cblas_dgbmv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const int KL, const int KU, const double alpha,
                             const double *A, const int lda, const double *X,
                             const int incX, const double beta, double *Y, const int incY);
    
extern void (*cblas_sgbmv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const int KL, const int KU, const float alpha,
                             const float *A, const int lda, const float *X,
                             const int incX, const float beta, float *Y, const int incY);
//GEMV
extern void (*cblas_zgemv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda,
                             const blasxDoubleComplex *X, const int incX, const blasxDoubleComplex *beta,
                             blasxDoubleComplex *Y, const int incY);
    
extern void (*cblas_cgemv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const void *alpha, const void *A, const int lda,
                             const void *X, const int incX, const void *beta,
                             void *Y, const int incY);
    
extern void (*cblas_sgemv_p)(const enum CBLAS_ORDER order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const float alpha, const float *A, const int lda,
                             const float *X, const int incX, const float beta,
                             float *Y, const int incY);
    
extern void (*cblas_dgemv_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                             const double alpha, const double *A, const int lda,
                             const double *X, const int incX, const double beta,
                             double *Y, const int incY);

/*******************LEVEL2-END*********************/

/*********************LEVEL3***********************/
//GEMM
extern void (*cblas_sgemm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_TRANSPOSE TransB,
                             const int M,
                             const int N,
                             const int K,
                             const float alpha,
                             const float *A,
                             const int lda,
                             const float *B,
                             const int ldb,
                             const float beta,
                             float *C,
                             const int ldc);

extern void (*cblas_dgemm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_TRANSPOSE TransB,
                             const int M,
                             const int N,
                             const int K,
                             const double alpha,
                             const double *A,
                             const int lda,
                             const double *B,
                             const int ldb,
                             const double beta,
                             double *C,
                             const int ldc );
    
extern void (*cblas_cgemm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_TRANSPOSE TransB,
                             const int M,
                             const int N,
                             const int K,
                             const blasxFloatComplex *alpha,
                             const blasxFloatComplex *A,
                             const int lda,
                             const blasxFloatComplex *B,
                             const int ldb,
                             const blasxFloatComplex *beta,
                             blasxFloatComplex *C,
                             const int ldc);
    
extern void (*cblas_zgemm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_TRANSPOSE TransB,
                             const int M,
                             const int N,
                             const int K,
                             const blasxDoubleComplex *alpha,
                             const blasxDoubleComplex *A,
                             const int lda,
                             const blasxDoubleComplex *B,
                             const int ldb,
                             const blasxDoubleComplex *beta,
                             blasxDoubleComplex *C,
                             const int ldc);
//SYMM
extern void (*cblas_ssymm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const int M,
                             const int N,
                             const float alpha,
                             const float *A,
                             const int lda,
                             const float *B,
                             const int ldb,
                             const float beta,
                             float *C,
                             const int ldc);

extern void (*cblas_dsymm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const int M,
                             const int N,
                             const double alpha,
                             const double *A,
                             const int lda,
                             const double *B,
                             const int ldb,
                             const double beta,
                             double *C,
                             const int ldc);
    
extern void (*cblas_csymm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const int M,
                             const int N,
                             const blasxFloatComplex *alpha,
                             const blasxFloatComplex *A,
                             const int lda,
                             const blasxFloatComplex *B,
                             const int ldb,
                             const blasxFloatComplex *beta,
                             blasxFloatComplex *C,
                             const int ldc);
    
extern void (*cblas_zsymm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const int M,
                             const int N,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             const void *B,
                             const int ldb,
                             const void *beta,
                             void *C,
                             const int ldc);
//SYRK
extern void (*cblas_zsyrk_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             const void *beta,
                             void *C,
                             const int ldc);
    
extern void (*cblas_csyrk_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const blasxFloatComplex *alpha,
                             const blasxFloatComplex *A,
                             const int lda,
                             const blasxFloatComplex *beta,
                             blasxFloatComplex *C,
                             const int ldc);
    
extern void (*cblas_dsyrk_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const double alpha,
                             const double *A,
                             const int lda,
                             const double beta,
                             double *C,
                             const int ldc);

extern void (*cblas_ssyrk_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const float alpha,
                             const float *A,
                             const int lda,
                             const float beta,
                             float *C,
                             const int ldc);

//SYR2K
extern void (*cblas_zsyr2k_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             const void *B,
                             const int ldb,
                             const void *beta,
                             void *C,
                             const int ldc);
    
extern void (*cblas_csyr2k_p)(const enum CBLAS_ORDER Order,
                              const enum CBLAS_UPLO Uplo,
                              const enum CBLAS_TRANSPOSE Trans,
                              const int N,
                              const int K,
                              const blasxFloatComplex *alpha,
                              const blasxFloatComplex *A,
                              const int lda,
                              const blasxFloatComplex *B,
                              const int ldb,
                              const blasxFloatComplex *beta,
                              blasxFloatComplex *C,
                              const int ldc);

extern void (*cblas_ssyr2k_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const float alpha,
                             const float *A,
                             const int lda,
                             const float *B,
                             const int ldb,
                             const float beta,
                             float *C,
                             const int ldc);
    
extern void (*cblas_dsyr2k_p)(const enum CBLAS_ORDER Order,
                              const enum CBLAS_UPLO Uplo,
                              const enum CBLAS_TRANSPOSE Trans,
                              const int N,
                              const int K,
                              const double alpha,
                              const double *A,
                              const int lda,
                              const double *B,
                              const int ldb,
                              const double beta,
                              double *C,
                              const int ldc);

    

//TRMM
extern void (*cblas_ztrmm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             void *B,
                             const int ldb);
extern void (*cblas_strmm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const float alpha,
                             const float *A,
                             const int lda,
                             float *B,
                             const int ldb);

extern void (*cblas_ctrmm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const blasxFloatComplex *alpha,
                             const blasxFloatComplex *A,
                             const int lda,
                             blasxFloatComplex *B,
                             const int ldb);


extern void (*cblas_dtrmm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const double alpha,
                             const double *A,
                             const int lda,
                             double *B,
                             const int ldb);

//TRSM
extern void (*cblas_ztrsm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             void *B,
                             const int ldb);

extern void (*cblas_ctrsm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const blasxFloatComplex *alpha,
                             const blasxFloatComplex *A,
                             const int lda,
                             blasxFloatComplex *B,
                             const int ldb);
    
extern void (*cblas_strsm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const float alpha,
                             const float *A,
                             const int lda,
                             float *B,
                             const int ldb);
extern void (*cblas_dtrsm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE TransA,
                             const enum CBLAS_DIAG Diag,
                             const int M,
                             const int N,
                             const double alpha,
                             const double *A,
                             const int lda,
                             double *B,
                             const int ldb);
    

//HEMM
extern void (*cblas_zhemm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const int M,
                             const int N,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             const void *B,
                             const int ldb,
                             const void *beta,
                             void *C,
                             const int ldc);
    
extern void (*cblas_chemm_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_SIDE Side,
                             const enum CBLAS_UPLO Uplo,
                             const int M,
                             const int N,
                             const void *alpha,
                             const void *A,
                             const int lda,
                             const void *B,
                             const int ldb,
                             const void *beta,
                             void *C,
                             const int ldc);

//HERK
extern void (*cblas_zherk_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const double alpha,
                             const void *A,
                             const int lda,
                             const double beta,
                             void *C,
                             const int ldc);
extern void (*cblas_cherk_p)(const enum CBLAS_ORDER Order,
                             const enum CBLAS_UPLO Uplo,
                             const enum CBLAS_TRANSPOSE Trans,
                             const int N,
                             const int K,
                             const float alpha,
                             const blasxFloatComplex *A,
                             const int lda,
                             const float beta,
                             blasxFloatComplex *C,
                             const int ldc);

//HER2K
extern void (*cblas_zher2k_p)(const enum CBLAS_ORDER Order,
                              const enum CBLAS_UPLO Uplo,
                              const enum CBLAS_TRANSPOSE Trans,
                              const int N,
                              const int K,
                              const void *alpha,
                              const void *A,
                              const int lda,
                              const void *B,
                              const int ldb,
                              const double beta,
                              void *C,
                              const int ldc);

extern void (*cblas_cher2k_p)(const enum CBLAS_ORDER Order,
                              const enum CBLAS_UPLO Uplo,
                              const enum CBLAS_TRANSPOSE Trans,
                              const int N,
                              const int K,
                              const void *alpha,
                              const blasxFloatComplex *A,
                              const int lda,
                              const void *B,
                              const int ldb,
                              const float beta,
                              blasxFloatComplex *C,
                              const int ldc);
/*********************LEVEL3***********************/


#ifdef __cplusplus
}
#endif

#endif /* BLASX_GLOBALPOINTERS */
