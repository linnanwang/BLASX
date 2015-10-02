#ifndef BLASX_INTERNAL_H
#define BLASX_INTERNAL_H
#include <dlfcn.h>
#include <blasx.h>
#include <cblas.h>
#include <sys/time.h>
#include <blasx_common.h>
#include <blasx_tile_resource.h>
//#define blasxDoubleComplex		void
//#define blasxDoubleComplexF77 	double
//#define blasxFloatComplex 		void
//#define blasxFloatComplexF77	float

#ifdef __cplusplus
extern "C" {
#endif
    inline void Blasx_Debug_Output(const char *fmt, ...);
    double get_cur_time();
    typedef enum { CPU, CUBLASXT, BLASX } blasx_operation_t;
    void blasx_init(const blasx_operation_t t);
    //CBLAS to CUBLAS
    inline int CBLasTransToCuBlasTrans(enum CBLAS_TRANSPOSE transa, cublasOperation_t *t);
    inline int CBlasSideToCuBlasSide(enum CBLAS_SIDE side,cublasSideMode_t *t);
    inline int CBlasFilledModeToCuBlasFilledMode(enum CBLAS_UPLO uplo, cublasFillMode_t *t);
    inline int CBlasDiagModeToCuBlasDiagMode(enum CBLAS_DIAG diag, cublasDiagType_t *t);
    //F77 to CBLAS parameters transformation
    inline int F77SideToCBlasSide(char *type, enum CBLAS_SIDE *side);
    inline int F77UploToCBlasUplo(char *type, enum CBLAS_UPLO *uplo);
    inline int F77TransToCBLASTrans(char *type, enum CBLAS_TRANSPOSE *trans);
    inline int F77DiagToCBLASDiag(char *type, enum CBLAS_DIAG *diag);

    
#ifdef __cplusplus
}
#endif

#endif /* BLASX_INTERNAL_H */
