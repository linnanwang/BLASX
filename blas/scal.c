#include <blasx.h>

void cblas_sscal(const int N, const float alpha, float *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sscal_p == NULL) blasx_init_cblas_func(&cblas_sscal_p, "cblas_sscal");
    Blasx_Debug_Output("Calling cblas_sscal\n ");
    (*cblas_sscal_p)(N,alpha,X,incX);
}
void sscal_(int *n, float *alpha, float *X,int *incx)
{
    Blasx_Debug_Output("Calling cblas_sscal\n ");
    cblas_sscal(*n,*alpha,X,*incx);
}

void cblas_dscal(const int N, const double alpha, double *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dscal_p == NULL) blasx_init_cblas_func(&cblas_dscal_p, "cblas_dscal");
    Blasx_Debug_Output("Calling cblas_dscal\n ");
    (*cblas_dscal_p)(N,alpha,X,incX);
}
void dscal_(int *n,double *alpha,double *X,int *incx)
{
    Blasx_Debug_Output("Calling dscal_\n ");
    cblas_dscal(*n,*alpha,X,*incx);
}

void cblas_cscal(const int N, const void *alpha, void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_cscal_p == NULL) blasx_init_cblas_func(&cblas_cscal_p, "cblas_cscal");
    Blasx_Debug_Output("Calling cblas_cscal\n ");
    (*cblas_cscal_p)(N,alpha,X,incX);
}
void cscal_(int *n,float *alpha,float *X,int *incx)
{
    Blasx_Debug_Output("Calling cscal_\n ");
    cblas_cscal(*n,alpha,X,*incx);
}

void cblas_zscal(const int N, const void *alpha, void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zscal_p == NULL) blasx_init_cblas_func(&cblas_zscal_p, "cblas_zscal");
    Blasx_Debug_Output("Calling cblas_zscal\n ");
    (*cblas_zscal_p)(N,alpha,X,incX);
}
void zscal_(int *n,double *alpha, double *X, int *incx)
{
    Blasx_Debug_Output("Calling zscal_\n ");
    cblas_zscal(*n,alpha,X,*incx);
}

void cblas_csscal(const int N, const float alpha, void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_csscal_p == NULL) blasx_init_cblas_func(&cblas_csscal_p, "cblas_csscal");
    Blasx_Debug_Output("Calling cblas_csscal_p\n ");
    (*cblas_csscal_p)(N,alpha,X,incX);
}
void csscal_(int *n,float *alpha, float *X, int *incx)
{
    Blasx_Debug_Output("Calling csscal_\n ");
    cblas_csscal(*n,*alpha,X,*incx);
}

void cblas_zdscal(const int N, const double alpha, void *X, const int incX)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zdscal_p == NULL) blasx_init_cblas_func(&cblas_zdscal_p, "cblas_zdscal");
    Blasx_Debug_Output("Calling cblas_zdscal_p\n ");
    (*cblas_zdscal_p)(N,alpha,X,incX);
}
void zdscal_(int *n,double *alpha,double *X,int *incx)
{
    Blasx_Debug_Output("Calling zdscal_\n ");
    cblas_zdscal(*n,*alpha,X,*incx);
}
