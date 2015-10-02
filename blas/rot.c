#include "blasx.h"
//ROT
void cblas_srot(const int N, float *X, const int incX,
                float *Y, const int incY, const float c, const float s)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_srot_p == NULL) blasx_init_cblas_func(&cblas_srot_p, "cblas_srot");
    Blasx_Debug_Output("Calling cblas_srot\n ");
    (*cblas_srot_p)(N,X,incX,Y,incY,c,s);
}

double dcabs1_(double *z) {
    return *z;
}

void srot_(int *n,float *X,int *incx, float *Y, int *incy, float *C, float *s)
{
    Blasx_Debug_Output("Calling srot_\n ");
    cblas_srot(*n,X,*incx,Y,*incy,*C,*s);
}

void cblas_drot(const int N, double *X, const int incX,
                double *Y, const int incY, const double c, const double  s)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_drot_p == NULL) blasx_init_cblas_func(&cblas_drot_p, "cblas_drot");
    Blasx_Debug_Output("Calling cblas_drot\n ");
    (*cblas_drot_p)(N,X,incX,Y,incY,c,s);
}

void drot_(int *n,double *X,int *incx, double *Y, int *incy, double *C, double *s)
{
    Blasx_Debug_Output("Calling drot_\n ");
    cblas_drot(*n,X,*incx,Y,*incy,*C,*s);
}

void cblas_csrot(const int N, void *X, const int incX,
                 void *Y, const int incY, const float c, const float s)
{
    float *x = X, *y = Y;
    int incx = incX, incy = incY;
    
    if (N > 0)
    {
        if (incX < 0)
        {
            if (incY < 0) { incx = -incx; incy = -incy; }
            else x += -incX * ((N-1)<<1);
        }
        else if (incY < 0)
        {
            incy = -incy;
            incx = -incx;
            x += (N-1)*(incX<<1);
        }
        if (csrot_p == NULL) blasx_init_cblas_func(&csrot_p, "csrot_");
        Blasx_Debug_Output("Calling cblas_csrot\n ");
        (*csrot_p)((int*)&N, x, &incx, y, &incy, (float*)&c, (float*)&s);
    }
}
void csrot_(int *n, float *X, int *incX, float *Y, int *incy, float *c, float *s)
{
    Blasx_Debug_Output("Calling csrot_\n ");
    cblas_csrot(*n,X,*incX,Y,*incy,*c,*s);
}

void cblas_zdrot(const int N, void *X, const int incX,
                 void *Y, const int incY, const double c, const double s)
{
    double *x = X, *y = Y;
    int incx = incX, incy = incY;
    
    if (N > 0)
    {
        if (incX < 0)
        {
            if (incY < 0) { incx = -incx; incy = -incy; }
            else x += -incX * ((N-1)<<1);
        }
        else if (incY < 0)
        {
            incy = -incy;
            incx = -incx;
            x += (N-1)*(incX<<1);
        }
        (*zdrot_p)((int*)&N, x, &incx, y, &incy, (double*)&c, (double*)&s);
    }
}
void zdrot_(int *n, float *X, int *incX, float *Y, int *incy, float *c, float *s)
{
    Blasx_Debug_Output("Calling zdrot_\n ");
    cblas_zdrot(*n,X,*incX,Y,*incy,*c,*s);
}


//ROTG
void cblas_srotg(float *a, float *b, float *c, float *s)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_srotg_p == NULL) blasx_init_cblas_func(&cblas_srotg_p, "cblas_srotg");
    Blasx_Debug_Output("Calling cblas_srotg\n ");
    (*cblas_srotg_p)(a,b,c,s);
}

void srotg_(float *a, float *b, float *c, float *s){
    Blasx_Debug_Output("Calling srotg_\n ");
    cblas_srotg(a,b,c,s);
}

void crotg_(float *a, float *b, float *c, float *s){
        Blasx_Debug_Output("Calling srotg_\n ");
            cblas_srotg(a,b,c,s);
}

void cblas_drotg(double *a, double *b, double *c, double *s){
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_drotg_p == NULL) blasx_init_cblas_func(&cblas_drotg_p, "cblas_drotg");
    Blasx_Debug_Output("Calling cblas_drotg\n ");
    (*cblas_drotg_p)(a,b,c,s);
}

void zrotg_(double *a, double *b, double *c, double *s){
        Blasx_Debug_Output("Calling drotg_\n ");
            cblas_drotg(a,b,c,s);
}

void drotg_(double *a, double *b, double *c, double *s){
    Blasx_Debug_Output("Calling drotg_\n ");
    cblas_drotg(a,b,c,s);
}

//ROTM
void cblas_srotm(const int N, float *X, const int incX,
                 float *Y, const int incY, const float *P)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_srotm_p == NULL) blasx_init_cblas_func(&cblas_srotm_p, "cblas_srotm");
    Blasx_Debug_Output("Calling cblas_srotm\n ");
    (*cblas_srotm_p)(N,X,incX,Y,incY,P);
}
void srotm_(int *n, float *X, int *incx, float *Y, int *incy, float *P)
{
    Blasx_Debug_Output("Calling srotm_\n ");
    cblas_srotm(*n,X,*incx,Y,*incy,P);
}

void cblas_drotm(const int N, double *X, const int incX,
                 double *Y, const int incY, const double *P)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_drotm_p == NULL) blasx_init_cblas_func(&cblas_drotm_p, "cblas_drotm");
    Blasx_Debug_Output("Calling cblas_drotm\n ");
    (*cblas_drotm_p)(N,X,incX,Y,incY,P);
}
void drotm_(int *n, double *X, int *incx, double *Y, int *incy, double *P)
{
    Blasx_Debug_Output("Calling drotm_\n ");
    cblas_drotm_p(*n,X,*incx,Y,*incy,P);
}
//ROTMG
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_srotmg_p == NULL) blasx_init_cblas_func(&cblas_srotmg_p, "cblas_srotmg");
    Blasx_Debug_Output("Calling cblas_srotmg\n ");
    (*cblas_srotmg_p)(d1,d2,b1,b2,P);
}
void srotmg_(float *d1, float *d2, float *b1, float *b2, float *P)
{
    Blasx_Debug_Output("Calling srotmg_\n ");
    cblas_srotmg(d1,d2,b1,*b2,P);
}

void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P)
{
    if (cpublas_handle == NULL) blasx_init(CPU);
    if (cblas_drotmg_p == NULL) blasx_init_cblas_func(&cblas_drotmg_p, "cblas_drotmg");
    Blasx_Debug_Output("Calling cblas_drotmg\n ");
    (*cblas_drotmg_p)(d1,d2,b1,b2,P);
}
void drotmg_(double *d1, double *d2, double *b1, double *b2, double *P)
{
    Blasx_Debug_Output("Calling drotmg_\n ");
    cblas_drotmg(d1,d2,b1,*b2,P);
}











