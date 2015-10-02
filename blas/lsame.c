#include "blasx.h"
#include <math.h>
#define ERROR_NAME "LSAME "

/* f77 interface */
int lsame_(char *a, char *b)
{	
	if(fabs(*a-*b) == 0 || fabs(*a-*b)==32){
		return 0;
	}else{
		return 1;
	}
}

