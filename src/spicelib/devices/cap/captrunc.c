/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "capdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CAPtrunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    CKTcircuit *ckt;
    double *timeStep;
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

    for( ; model!= NULL; model = model->CAPnextModel) {
        for(here = model->CAPinstances ; here != NULL ;
                here = here->CAPnextInstance) {
	    if (here->CAPowner != ARCHme) continue;

            CKTterr(here->CAPqcap,ckt,timeStep);
        }
    }
    return(OK);
}
