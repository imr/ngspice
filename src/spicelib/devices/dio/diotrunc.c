/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


int
DIOtrunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    double *timeStep;
{
    register DIOmodel *model = (DIOmodel*)inModel;
    register DIOinstance *here;

    for( ; model != NULL; model = model->DIOnextModel) {
        for(here=model->DIOinstances;here!=NULL;here = here->DIOnextInstance){
	    if (here->DIOowner != ARCHme) continue;
            CKTterr(here->DIOcapCharge,ckt,timeStep);
        }
    }
    return(OK);
}
