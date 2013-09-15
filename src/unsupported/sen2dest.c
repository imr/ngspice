/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* SENdestroy(ckt)
 * this is a driver program to iterate through all the various
 * destroy functions provided for the circuit elements in the
 * given circuit
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void
SENdestroy(SENstruct *info)
{
    int i;
    int size;

    size = info->SENsize;

#ifdef SENSDEBUG
    printf("size = %d\n", size);
    printf("freeing sensitivity structure in SENdestroy\n");
#endif

    /*
      if (info->SENdevices) FREE(info->SENdevices);
      if (info->SENparmNames) FREE(info->SENparmNames);
    */

    if (info->SEN_Sap) {

#ifdef SENSDEBUG
        printf("freeing SEN_Sap in SENdestroy\n");
#endif

        for (i = 0; i <= size; i++)
            if (info->SEN_Sap[i])
                FREE(info->SEN_Sap[i]);
        FREE(info->SEN_Sap);
    }

    if (info->SEN_RHS) {
        for (i = 0; i <= size; i++)
            if (info->SEN_RHS[i])
                FREE(info->SEN_RHS[i]);
        FREE(info->SEN_RHS);
    }

    if (info->SEN_iRHS) {
        for (i = 0; i <= size; i++)
            if (info->SEN_iRHS[i])
                FREE(info->SEN_iRHS[i]);
        FREE(info->SEN_Sap);
    }

    /*
      FREE(info);
    */

#ifdef SENSDEBUG
    printf("SENdestroy end\n");
#endif

    return;
}
