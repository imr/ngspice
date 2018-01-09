/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;

    for( ; model != NULL; model = DIOnextModel(model)) {
        for(here=DIOinstances(model);here!=NULL;here = DIOnextInstance(here)){
            CKTterr(here->DIOcapCharge,ckt,timeStep);
        }
    }
    return(OK);
}
