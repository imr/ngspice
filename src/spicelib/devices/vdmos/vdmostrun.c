/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VDMOStrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;

    for( ; model != NULL; model = VDMOSnextModel(model)) {
        for(here=VDMOSinstances(model);here!=NULL;here = VDMOSnextInstance(here)){
        
            CKTterr(here->VDMOSqgs,ckt,timeStep);
            CKTterr(here->VDMOSqgd,ckt,timeStep);
            CKTterr(here->VDIOcapCharge,ckt,timeStep);
        }
    }
    return(OK);
}
