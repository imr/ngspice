#include <stdio.h>
#include "ngspice/sperror.h"

extern "C"
__device__
static
int
cuNIintegrate_device_kernel
(
double *CKTstate_0, double *CKTstate_1, double *geq, double *ceq,
double value, int charge, double CKTag_0, double CKTag_1, int CKTorder
)
{

#define current charge+1

    switch (CKTorder)
    {
        case 1:
            CKTstate_0 [current] = CKTag_0 * (CKTstate_0 [charge]) + CKTag_1 * (CKTstate_1 [charge]) ;
            break ;

        case 2:
            CKTstate_0 [current] = -CKTstate_1 [current] * CKTag_1 + CKTag_0 * (CKTstate_0 [charge] - CKTstate_1 [charge]) ;
            break ;

        default:
            printf ("Error inside the integration formula\n") ;
            return (E_ORDER) ;
    }

    *ceq = CKTstate_0 [current] - CKTag_0 * CKTstate_0 [charge] ;
    *geq = CKTag_0 * value ;

    return (OK) ;
}
