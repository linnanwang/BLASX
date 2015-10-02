#include <stdio.h>
#include <error.h>
#include <stdlib.h>

void xerbla_(char *message, int *info){
    printf(" ** On entry to %6s parameter number %2d had an illegal value\n",
           message, *info);
}

/*-------------blas error handler-----------------*/
void BlasX_Check_Status(cublasStatus_t status){
    if(status==CUBLAS_STATUS_NOT_INITIALIZED){
        printf("The CUBLAS library was not initialized.\n");
    }
    else if(status==CUBLAS_STATUS_ALLOC_FAILED){
        printf("Resource allocation failed inside the CUBLAS library.\n");
    }else if(status==CUBLAS_STATUS_INVALID_VALUE){
        printf("An unsupported value or parameter was passed to the function.\n");
    }else if(status==CUBLAS_STATUS_ARCH_MISMATCH){
        printf("The function requires a feature absent from the device architecture.\n");
    }else if(status==CUBLAS_STATUS_MAPPING_ERROR){
        printf("An access to GPU memory space failed.\n");
    }else if(status==CUBLAS_STATUS_EXECUTION_FAILED){
        printf("The GPU program failed to execute.\n");
    }else if(status==CUBLAS_STATUS_INTERNAL_ERROR){
        printf("An internal CUBLAS operation failed.\n");
    }
}





