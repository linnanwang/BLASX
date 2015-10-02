#include <blasx_globalpointers.h>


/*----- BLAS Functions----*/
/*********************LEVEL1***********************/
//AMAX
size_t (*cblas_isamax_p)(const int N, const float  *X, const int incX) = NULL;
size_t (*cblas_idamax_p)(const int N, const double *X, const int incX) = NULL;
size_t (*cblas_icamax_p)(const int N, const void   *X, const int incX) = NULL;
size_t (*cblas_izamax_p)(const int N, const void   *X, const int incX) = NULL;

//SWAP
void (*cblas_sswap_p)(const int N, float *X, const int incX,
                      float *Y, const int incY) = NULL;
void (*cblas_dswap_p)(const int N, double *X, const int incX,
                      double *Y, const int incY) = NULL;
void (*cblas_cswap_p)(const int N, void *X, const int incX,
                      void *Y, const int incY) = NULL;
void (*cblas_zswap_p)(const int N, void *X, const int incX,
                      void *Y, const int incY) = NULL;

//SCAL
void (*cblas_sscal_p)(const int N, const float alpha, float *X, const int incX) = NULL;
void (*cblas_dscal_p)(const int N, const double alpha, double *X, const int incX) = NULL;
void (*cblas_cscal_p)(const int N, const void *alpha, void *X, const int incX) = NULL;
void (*cblas_zscal_p)(const int N, const void *alpha, void *X, const int incX) = NULL;
void (*cblas_csscal_p)(const int N, const float alpha, void *X, const int incX) = NULL;
void (*cblas_zdscal_p)(const int N, const double alpha, void *X, const int incX) = NULL;

//ROTMG
void (*cblas_drotmg_p)(double *d1, double *d2, double *b1, const double b2, double *P) = NULL;
void (*cblas_srotmg_p)(float *d1, float *d2, float *b1, const float b2, float *P) = NULL;


//ROTM
void (*cblas_srotm_p)(const int N, float *X, const int incX,
                      float *Y, const int incY, const float *P) = NULL;
void (*cblas_drotm_p)(const int N, double *X, const int incX,
                      double *Y, const int incY, const double *P) = NULL;

//ROTG
void (*cblas_srotg_p)(float *a, float *b, float *c, float *s) = NULL;
void (*cblas_drotg_p)(double *a, double *b, double *c, double *s) = NULL;

//ROT
void (*cblas_srot_p)(const int N, float *X, const int incX,
                     float *Y, const int incY, const float c, const float s) = NULL;
void (*cblas_drot_p)(const int N, double *X, const int incX,
                     double *Y, const int incY, const double c, const double s) = NULL;
void (*csrot_p)(int *n, float *X, int *incX, float *Y, int *incy, float *C, float *s) = NULL;
void (*zdrot_p)(int *n, double *X, int *incX, double *Y, int *incy, double *c, double *s) = NULL;


//NRM2
float (*cblas_snrm2_p)(const int N, const float *X, const int incX) = NULL;
double (*cblas_dnrm2_p)(const int N, const double *X, const int incX) = NULL;
float (*cblas_scnrm2_p)(const int N, const void *X, const int incX) = NULL;
double (*cblas_dznrm2_p)(const int N, const void *X, const int incX) = NULL;

//DOTU
void (*cblas_zdotu_sub_p)(const int N, const void *X, const int incX,
                          const void *Y, const int incY, void *dotu) = NULL;
void (*cblas_cdotu_sub_p)(const int N, const void *X, const int incX,
                          const void *Y, const int incY, void *dotu) = NULL;
//DOTC
void (*cblas_zdotc_sub_p)(const int N, const void *X, const int incX,
                          const void *Y, const int incY, void *dotc) = NULL;
void (*cblas_cdotc_sub_p)(const int N, const void *X, const int incX,
                          const void *Y, const int incY, void *dotc) = NULL;

//SDOT
float (*cblas_sdsdot_p)(const int N, const float alpha, const float *X,
                        const int incX, const float *Y, const int incY) = NULL;
double (*cblas_dsdot_p)(const int N, const float *X, const int incX, const float *Y,
                        const int incY) = NULL;

//DOT
float (*cblas_sdot_p)(const int N, const float  *X, const int incX,
                             const float  *Y, const int incY) = NULL;
double (*cblas_ddot_p)(const int N, const double *X, const int incX,
                              const double *Y, const int incY) = NULL;

//COPY
void (*cblas_scopy_p)(const int N, const float *X, const int incX,
                      float *Y, const int incY) = NULL;
void (*cblas_dcopy_p)(const int N, const double *X, const int incX,
                      double *Y, const int incY) = NULL;
void (*cblas_ccopy_p)(const int N, const void *X, const int incX,
                      void *Y, const int incY) = NULL;
void (*cblas_zcopy_p)(const int N, const void *X, const int incX,
                      void *Y, const int incY) = NULL;

//AXPY
void (*cblas_saxpy_p)(const int N, const float alpha, const float *X,
                      const int incX, float *Y, const int incY) = NULL;
void (*cblas_daxpy_p)(const int N, const double alpha, const double *X,
                      const int incX, double *Y, const int incY) = NULL;
void (*cblas_caxpy_p)(const int N, const void *alpha, const void *X,
                      const int incX, void *Y, const int incY) = NULL;
void (*cblas_zaxpy_p)(const int N, const void *alpha, const void *X,
                      const int incX, void *Y, const int incY) = NULL;

//ASUM
float (*cblas_sasum_p)(const int N, const float *X, const int incX) = NULL;
double (*cblas_dasum_p)(const int N, const double *X, const int incX) = NULL;
double (*cblas_dzasum_p)(const int N, const void *X, const int incX) = NULL;
float (*cblas_scasum_p)(const int N, const void *X, const int incX) = NULL;


/*********************LEVEL2***********************/
//SYR2
void (*cblas_dsyr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const double alpha, const double *X,
                      const int incX, const double *Y, const int incY, double *A,
                      const int lda) = NULL;

void (*cblas_ssyr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const float alpha, const float *X,
                      const int incX, const float *Y, const int incY, float *A,
                      const int lda) = NULL;
//SYR
void (*cblas_dsyr_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const int N, const double alpha, const double *X,
                     const int incX, double *A, const int lda) = NULL;

void (*cblas_ssyr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const float alpha, const float *X,
                     const int incX, float *A, const int lda) = NULL;


//SYMV
void (*cblas_dsymv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const double alpha, const double *A,
                      const int lda, const double *X, const int incX,
                      const double beta, double *Y, const int incY) = NULL;

void (*cblas_ssymv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const float alpha, const float *A,
                      const int lda, const float *X, const int incX,
                      const float beta, float *Y, const int incY) = NULL;
//SPR2
void (*cblas_dspr2_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const int N, const double alpha, const double *X,
                      const int incX, const double *Y, const int incY, double *A) = NULL;

void (*cblas_sspr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const float alpha, const float *X,
                      const int incX, const float *Y, const int incY, float *A) = NULL;

//SPR
void (*cblas_dspr_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const int N, const double alpha, const double *X,
                     const int incX, double *Ap) = NULL;

void (*cblas_sspr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const float alpha, const float *X,
                     const int incX, float *Ap) = NULL;

//SPMV
void (*cblas_dspmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const double alpha, const double *Ap,
                      const double *X, const int incX,
                      const double beta, double *Y, const int incY) = NULL;

void (*cblas_sspmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const float alpha, const float *Ap,
                      const float *X, const int incX,
                      const float beta, float *Y, const int incY) = NULL;

//SBMV
void (*cblas_dsbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const int K, const double alpha, const double *A,
                      const int lda, const double *X, const int incX,
                      const double beta, double *Y, const int incY) = NULL;

void (*cblas_ssbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                             const int N, const int K, const float alpha, const float *A,
                             const int lda, const float *X, const int incX,
                             const float beta, float *Y, const int incY);


//GER
void (*cblas_dger_p)(const enum CBLAS_ORDER order, const int M, const int N,
                     const double alpha, const double *X, const int incX,
                     const double *Y, const int incY, double *A, const int lda) = NULL;

void (*cblas_sger_p)(const enum CBLAS_ORDER order, const int M, const int N,
                     const float alpha, const float *X, const int incX,
                     const float *Y, const int incY, float *A, const int lda) = NULL;


//TRSV
void (*cblas_ztrsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *A, const int lda, void *X,
                      const int incX) = NULL;

void (*cblas_ctrsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *A, const int lda, void *X,
                      const int incX) = NULL;

void (*cblas_dtrsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const double *A, const int lda, double *X,
                      const int incX) = NULL;

void (*cblas_strsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const float *A, const int lda, float *X,
                      const int incX);

//TRMV
void (*cblas_ztrmv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *A, const int lda,
                      void *X, const int incX) = NULL;

void (*cblas_ctrmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *A, const int lda,
                      void *X, const int incX) = NULL;

void (*cblas_dtrmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const double *A, const int lda,
                      double *X, const int incX) = NULL;

void (*cblas_strmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const float *A, const int lda,
                      float *X, const int incX) = NULL;


//TPSV
void (*cblas_ztpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *Ap, void *X, const int incX) = NULL;

void (*cblas_ctpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *Ap, void *X, const int incX) = NULL;

void (*cblas_dtpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const double *Ap, double *X, const int incX) = NULL;

void (*cblas_stpsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const float *Ap, float *X, const int incX) = NULL;


//TPMV
void (*cblas_ztpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *Ap, void *X, const int incX) = NULL;

void (*cblas_ctpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const void *Ap, void *X, const int incX) = NULL;

void (*cblas_dtpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const double *Ap, double *X, const int incX) = NULL;

void (*cblas_stpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const float *Ap, float *X, const int incX) = NULL;

//TBSV
void (*cblas_ztbsv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const void *A, const int lda,
                      void *X, const int incX) = NULL;

void (*cblas_ctbsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const void *A, const int lda,
                      void *X, const int incX) = NULL;


void (*cblas_dtbsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const double *A, const int lda,
                      double *X, const int incX) = NULL;

void (*cblas_stbsv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const float *A, const int lda,
                      float *X, const int incX) = NULL;


//TBMV
void (*cblas_ztbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const void *A, const int lda,
                      void *X, const int incX) = NULL;

void (*cblas_ctbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const void *A, const int lda,
                      void *X, const int incX) = NULL;


void (*cblas_dtbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const double *A, const int lda,
                      double *X, const int incX) = NULL;

void (*cblas_stbmv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int N, const int K, const float *A, const int lda,
                      float *X, const int incX) = NULL;

//HPR2
void (*cblas_zhpr2_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *Ap) = NULL;

void (*cblas_chpr2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *Ap) = NULL;

//HPR
void (*cblas_zhpr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const double alpha, const void *X,
                     const int incX, void *A) = NULL;

void (*cblas_chpr_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const float alpha, const void *X,
                     const int incX, void *A) = NULL;
//HPMV
void (*cblas_zhpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const void *alpha, const void *Ap,
                      const void *X, const int incX,
                      const void *beta, void *Y, const int incY) = NULL;

void (*cblas_chpmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const void *alpha, const void *Ap,
                      const void *X, const int incX,
                      const void *beta, void *Y, const int incY) = NULL;


//HER2
void (*cblas_zher2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *A, const int lda) = NULL;

void (*cblas_cher2_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *A, const int lda) = NULL;

//HER
void (*cblas_zher_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const int N, const double alpha, const void *X, const int incX,
                     void *A, const int ldA) = NULL;

void (*cblas_cher_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const float alpha, const void *X, const int incX,
                     void *A, const int lda) = NULL;

//HEMV
void (*cblas_zhemv_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const int N, const void *alpha, const void *A,
                      const int ldA, const void *X, const int incX,
                      const void *beta, void *Y, const int incY) = NULL;

void (*cblas_chemv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const void *alpha, const void *A,
                      const int lda, const void *X, const int incX,
                      const void *beta, void *Y, const int incY) = NULL;

//HBMV
void (*cblas_zhbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const int K, const void *alpha, const void *A,
                      const int lda, const void *X, const int incX,
                      const void *beta, void *Y, const int incY) = NULL;

void (*cblas_chbmv_p)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const int N, const int K, const void *alpha, const void *A,
                      const int lda, const void *X, const int incX,
                      const void *beta, void *Y, const int incY) = NULL;
//GERU
void (*cblas_zgeru_p)(const enum CBLAS_ORDER order, const int M, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *A, const int lda) = NULL;

void (*cblas_cgeru_p)(const enum CBLAS_ORDER order, const int M, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *A, const int lda) = NULL;

//GERC
void (*cblas_zgerc_p)(const enum CBLAS_ORDER Order, const int M, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *A, const int lda) = NULL;

void (*cblas_cgerc_p)(const enum CBLAS_ORDER order, const int M, const int N,
                      const void *alpha, const void *X, const int incX,
                      const void *Y, const int incY, void *A, const int lda) = NULL;

//GBMV
void (*cblas_zgbmv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const int KL, const int KU, const void *alpha,
                      const void *A, const int lda, const void *X,
                      const int incX, const void *beta, void *Y, const int incY) = NULL;

void (*cblas_cgbmv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const int KL, const int KU, const void *alpha,
                      const void *A, const int lda, const void *X,
                      const int incX, const void *beta, void *Y, const int incY) = NULL;

void (*cblas_dgbmv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const int KL, const int KU, const double alpha,
                      const double *A, const int lda, const double *X,
                      const int incX, const double beta, double *Y, const int incY) = NULL;

void (*cblas_sgbmv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const int KL, const int KU, const float alpha,
                      const float *A, const int lda, const float *X,
                      const int incX, const float beta, float *Y, const int incY) = NULL;

//GEMV
void (*cblas_zgemv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda,
                      const blasxDoubleComplex *X, const int incX, const blasxDoubleComplex *beta,
                      blasxDoubleComplex *Y, const int incY) = NULL;
void (*cblas_cgemv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const void *alpha, const void *A, const int lda,
                      const void *X, const int incX, const void *beta,
                      void *Y, const int incY) = NULL;
void (*cblas_sgemv_p)(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const float alpha, const float *A, const int lda,
                      const float *X, const int incX, const float beta,
                      float *Y, const int incY) = NULL;
void (*cblas_dgemv_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                      const double alpha, const double *A, const int lda,
                      const double *X, const int incX, const double beta,
                      double *Y, const int incY) = NULL;

/*********************LEVEL3***********************/
//GEMM
void (*cblas_sgemm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const float alpha, const float *A, const int lda,
                      const float *B, const int ldb, const float beta,
                      float *C, const int ldc
                      ) = NULL;

void (*cblas_dgemm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const double alpha, const double *A, const int lda,
                      const double *B, const int ldb, const double beta,
                      double *C, const int ldc
                      ) = NULL;

void (*cblas_cgemm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                       const int M, const int N, const int K,
                       const blasxFloatComplex *alpha, const blasxFloatComplex *A, const int lda,
                       const blasxFloatComplex *B, const int ldb, const blasxFloatComplex *beta,
                       blasxFloatComplex *C, const int ldc
                       ) = NULL;

void (*cblas_zgemm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda,
                      const blasxDoubleComplex *B, const int ldb, const blasxDoubleComplex *beta,
                      blasxDoubleComplex *C, const int ldc
                      ) = NULL;

//SYMM
void (*cblas_ssymm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const int M, const int N,
                      const float alpha, const float *A, const int lda,
                      const float *B, const int ldb, const float beta,
                      float *C,const int ldc
                      ) = NULL;

void (*cblas_dsymm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const int M, const int N,
                      const double alpha, const double *A, const int lda,
                      const double *B, const int ldb, const double beta,
                      double *C, const int ldc
                      ) = NULL;

void (*cblas_csymm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const int M, const int N,
                      const blasxFloatComplex *alpha, const blasxFloatComplex *A, const int lda,
                      const blasxFloatComplex *B, const int ldb,
                      const blasxFloatComplex *beta, blasxFloatComplex *C,
                      const int ldc) = NULL;

void (*cblas_zsymm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const int M, const int N,
                      const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda,
                      const blasxDoubleComplex *B, const int ldb,
                      const blasxDoubleComplex *beta, blasxDoubleComplex *C, const int ldc
                      ) = NULL;

//SYRK
void (*cblas_ssyrk_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                      const int N, const int K,
                      const float alpha, const float *A, const int lda,
                      const float beta, float *C, const int ldc
                      ) = NULL;

void (*cblas_dsyrk_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                      const int N, const int K,
                      const double alpha, const double *A, const int lda,
                      const double beta, double *C, const int ldc
                      ) = NULL;

void (*cblas_csyrk_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                        const int N, const int K,
                        const blasxFloatComplex *alpha, const blasxFloatComplex *A, const int lda,
                        const blasxFloatComplex *beta, blasxFloatComplex *C, const int ldc
                      ) = NULL;

void (*cblas_zsyrk_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                      const int N, const int K,
                      const blasxDoubleComplex *alpha, const blasxDoubleComplex *A, const int lda,
                      const blasxDoubleComplex *beta, blasxDoubleComplex *C, const int ldc
                      ) = NULL;

//SYR2K
void (*cblas_ssyr2k_p)(const enum CBLAS_ORDER Order,
                       const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                       const int N, const int K,
                       const float alpha, const float *A, const int lda,
                       const float *B, const int ldb,
                       const float beta, float *C, const int ldc
                       ) = NULL;

void (*cblas_dsyr2k_p)(const enum CBLAS_ORDER Order,
                       const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                       const int N, const int K,
                       const double alpha, const double *A, const int lda,
                       const double *B, const int ldb,
                       const double beta, double *C, const int ldc
                       ) = NULL;

void (*cblas_csyr2k_p)(const enum CBLAS_ORDER Order,
                       const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                       const int N, const int K,
                       const blasxFloatComplex *alpha, const blasxFloatComplex *A, const int lda,
                       const blasxFloatComplex *B, const int ldb,
                       const blasxFloatComplex *beta,
                       blasxFloatComplex *C, const int ldc
                       ) = NULL;

void (*cblas_zsyr2k_p)(const enum CBLAS_ORDER Order,
                       const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                       const int N, const int K,
                       const void *alpha, const void *A, const int lda,
                       const void *B, const int ldb,
                       const void *beta, void *C, const int ldc
                       ) = NULL;

//TRMM
void (*cblas_strmm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const float alpha, const float *A, const int lda,
                      float *B, const int ldb
                      ) = NULL;

void (*cblas_dtrmm_p)(const enum CBLAS_ORDER Order, 
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, 
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const double alpha, const double *A, const int lda,
                      double *B, const int ldb
                      ) = NULL;

void (*cblas_ctrmm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const blasxFloatComplex *alpha, const blasxFloatComplex *A,const int lda,
                      blasxFloatComplex *B, const int ldb
                      ) = NULL;

void (*cblas_ztrmm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const void *alpha, const void *A, const int lda,
                      void *B, const int ldb
                      ) = NULL;

//TRSM
void (*cblas_strsm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const float alpha, const float *A, const int lda,
                      float *B, const int ldb
                      ) = NULL;

void (*cblas_dtrsm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const double alpha, const double *A, const int lda,
                      double *B, const int ldb
                      ) = NULL;

void (*cblas_ctrsm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                      const int M, const int N,
                      const blasxFloatComplex *alpha, const blasxFloatComplex *A,const int lda,
                      blasxFloatComplex *B, const int ldb
                      ) = NULL;
					  
void (*cblas_ztrsm_p)(const enum CBLAS_ORDER Order,
					  const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
					  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
					  const int M, const int N,
					  const void *alpha, const void *A, const int lda,
					  void *B, const int ldb
					  )= NULL;

//HEMM
void (*cblas_zhemm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const int M, const int N,
                      const void *alpha, const void *A, const int lda,
                      const void *B, const int ldb, const void *beta,
                      void *C, const int ldc
                      ) = NULL;

void (*cblas_chemm_p)(const enum CBLAS_ORDER Order,
                      const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                      const int M, const int N,
                      const void *alpha, const void *A, const int lda,
                      const void *B, const int ldb, const void *beta,
                      void *C, const int ldc
                      ) = NULL;


//HERK
void (*cblas_zherk_p)(const enum CBLAS_ORDER Order,const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const double alpha, const void *A, const int lda,
                      const double beta, void *C, const int ldc
                      ) = NULL;

void (*cblas_cherk_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const float alpha, const blasxFloatComplex *A, const int lda,
                      const float beta, blasxFloatComplex *C,const int ldc
                      ) = NULL;

//HER2K
void (*cblas_zher2k_p)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                       const enum CBLAS_TRANSPOSE Trans,const int N,const int K,
                       const void *alpha,const void *A,const int lda,
                       const void *B,const int ldb,const double beta,
                       void *C,const int ldc
                       ) = NULL;

void (*cblas_cher2k_p)(const enum CBLAS_ORDER Order,const enum CBLAS_UPLO Uplo,
                       const enum CBLAS_TRANSPOSE Trans,const int N,const int K,
                       const void *alpha,const blasxFloatComplex *A,const int lda,
                       const void *B,const int ldb,const float beta,
                       blasxFloatComplex *C,const int ldc
                       ) = NULL;
