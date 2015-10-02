#include <blasx.h>
#define ERROR_NAME "AMAX "

size_t cblas_isamax(const int N, const float  *X, const int incX)
{
    int info = 0;
    if (incX < 0)                                         info = 3;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_isamax_p == NULL) blasx_init_cblas_func(&cblas_isamax_p, "cblas_isamax");
    Blasx_Debug_Output("Calling cblas_isamax, result is:%d, N:%d\n ",N);
    return (*cblas_isamax_p)(N,X,incX);
}
int isamax_(int *n, float *X,int *incx)
{
    Blasx_Debug_Output("Calling isamax_\n ");
    int max = cblas_isamax(*n,X,*incx);
    return *n?++max:max;
}

size_t cblas_idamax(const int N, const double *X, const int incX)
{
    int info = 0;
    if (incX < 0)                                         info = 3;
    if (info != 0) {
        xerbla_(ERROR_NAME, &info);
        return;
    }
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_idamax_p == NULL) blasx_init_cblas_func(&cblas_idamax_p, "cblas_idamax");
    Blasx_Debug_Output("Calling cblas_idamax\n ");
    return (*cblas_idamax_p)(N,X,incX);
}
int idamax_(int *n, double *X, int *incx)
{
    Blasx_Debug_Output("Calling idamax_\n ");
    int max = cblas_idamax(*n,X,*incx);
    return *n?++max:max;
}

size_t cblas_icamax(const int N, const void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_icamax_p == NULL) blasx_init_cblas_func(&cblas_icamax_p, "cblas_icamax");
    Blasx_Debug_Output("Calling cblas_icamax\n ");
    return (*cblas_icamax_p)(N,X,incX);
}
int icamax_(int *n, float *X, int *incx)
{
    Blasx_Debug_Output("Calling icamax_\n ");
    int max = cblas_icamax(*n,X,*incx);
    return *n?++max:max;
}

size_t cblas_izamax(const int N, const void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_izamax_p == NULL) blasx_init_cblas_func(&cblas_izamax_p, "cblas_izamax");
    Blasx_Debug_Output("Calling cblas_izamax\n ");
    return (*cblas_izamax_p)(N,X,incX);
}
int izamax_(int *n, double *X,int *incx)
{
    Blasx_Debug_Output("Calling izamax_\n ");
    int max = cblas_izamax(*n,X,*incx);
    return *n?++max:max;
}
