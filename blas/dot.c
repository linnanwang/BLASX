#include "blasx.h"
//DOT
float cblas_sdot(const int N, const float  *X, const int incX,
                 const float  *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sdot_p == NULL) blasx_init_cblas_func(&cblas_sdot_p, "cblas_sdot");
    Blasx_Debug_Output("Calling cblas_sdot\n ");
    return (*cblas_sdot_p)(N,X,incX,Y,incY);
}

float sdot_(int *n, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling sdot_\n ");
    return cblas_sdot(*n,X,*incx,Y,*incy);
}


double cblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_ddot_p == NULL) blasx_init_cblas_func(&cblas_ddot_p, "cblas_ddot");
    Blasx_Debug_Output("Calling cblas_ddot\n ");
    return (*cblas_ddot_p)(N,X,incX,Y,incY);
}

double ddot_(int *n, double *X, int *incx, double *Y, int *incy){
    Blasx_Debug_Output("Calling ddot_\n ");
    return cblas_ddot(*n,X,*incx,Y,*incy);
}

double cblas_dsdot(const int N, const float *X, const int incX, const float *Y,
                   const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_dsdot_p == NULL) blasx_init_cblas_func(&cblas_dsdot_p, "cblas_dsdot");
    Blasx_Debug_Output("Calling cblas_dsdot\n ");
    return (*cblas_dsdot_p)(N,X,incX,Y,incY);
}

double dsdot_(int *n, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling dsdot_\n ");
    return cblas_dsdot(*n,X,*incx,Y,*incy);
}


float cblas_sdsdot(const int N, const float alpha, const float *X,
                   const int incX, const float *Y, const int incY){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_sdsdot_p == NULL) blasx_init_cblas_func(&cblas_sdsdot_p, "cblas_sdsdot");
    Blasx_Debug_Output("Calling cblas_sdsdot\n ");
    return (*cblas_sdsdot_p)(N,alpha,X,incX,Y,incY);
}

double sdsdot_(int *n, float *alpha, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling sdsdot_\n ");
    return cblas_sdsdot(*n, *alpha,X,*incx,Y,*incy);
}
//DOTC
void cblas_zdotc_sub(const int N, const void *X, const int incX,
                     const void *Y, const int incY, void *dotc)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zdotc_sub_p == NULL) blasx_init_cblas_func(&cblas_zdotc_sub_p, "cblas_zdotc_sub");
    Blasx_Debug_Output("Calling cblas_zdotc_sub\n ");
    (*cblas_zdotc_sub_p)(N,X,incX,Y,incY,dotc);
}

blasx_complex_double zdotc_(int *n, double *X,int *incx,double *Y,int *incy)
{
    Blasx_Debug_Output("Calling zdotc_\n ");
    blasx_complex_double result;
    cblas_zdotc_sub(*n,X,*incx,Y,*incy,&result);
    return result;
}

void cblas_cdotc_sub(const int N, const void *X, const int incX,
                          const void *Y, const int incY, void *dotc)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_cdotc_sub_p == NULL) blasx_init_cblas_func(&cblas_cdotc_sub_p, "cblas_cdotc_sub");
    Blasx_Debug_Output("Calling cblas_cdotc_sub\n ");
    (*cblas_cdotc_sub_p)(N,X,incX,Y,incY,dotc);
}

blasx_complex_float cdotc_(int *n, float *X,int *incx,float *Y,int *incy){
    Blasx_Debug_Output("Calling cdotc\n ");
    blasx_complex_float result;
    cblas_cdotc_sub(*n,X,*incx,Y,*incy,&result);
    return result;
}

//DOTU
void cblas_cdotu_sub(const int N, const void *X, const int incX,
                     const void *Y, const int incY, void *dotu){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_cdotu_sub_p == NULL) blasx_init_cblas_func(&cblas_cdotu_sub_p, "cblas_cdotu_sub");
    Blasx_Debug_Output("Calling cblas_cdotu_sub\n ");
    (*cblas_cdotu_sub_p)(N,X,incX,Y,incY,dotu);
}

blasx_complex_float cdotu_(int *n, float *X, int *incx, float *Y, int *incy){
    Blasx_Debug_Output("Calling cdotu\n ");
    blasx_complex_float result;
    cblas_cdotu_sub(*n,X,*incx,Y,*incy,&result);
    return result;
}

void cblas_zdotu_sub(const int N, const void *X, const int incX,
                     const void *Y, const int incY, void *dotu)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_zdotu_sub_p == NULL) blasx_init_cblas_func(&cblas_zdotu_sub_p, "cblas_zdotu_sub");
    Blasx_Debug_Output("Calling cblas_zdotu_sub\n ");
    (*cblas_zdotu_sub_p)(N,X,incX,Y,incY,dotu);
}

blasx_complex_double zdotu_(int *n, double *X,int *incx,double *Y, int *incy)
{
    Blasx_Debug_Output("Calling zdotu_\n ");
    blasx_complex_double result;
    cblas_zdotu_sub(*n,X,*incx,Y,*incy,&result);
    return result;
}













