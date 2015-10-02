#ifndef BLASXERROR_H
#define BLASXERROR_H
#include <cublasXt.h>

enum BlasXError {
    BlasXSuccess = 0,
    BlasXError_Matrix_MNotCorrect = 1001,
    BlasXError_Matrix_NNotCorrect = 1002,
    BlasXError_Matrix_KNotCorrect = 1003,
};

void BlasX_Check_Status(cublasStatus_t status);
//error handler
void xerbla_(char *message, int *info);

#endif /* QUANTXERROR_H */

