/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "mos3defs.h"
#include "sperror.h"
#include "suffix.h"

int
MOS3trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;

    for( ; model != NULL; model = model->MOS3nextModel) {
        for(here=model->MOS3instances;here!=NULL;here = here->MOS3nextInstance){
	    if (here->MOS3owner != ARCHme) continue;
        
            CKTterr(here->MOS3qgs,ckt,timeStep);
            CKTterr(here->MOS3qgd,ckt,timeStep);
            CKTterr(here->MOS3qgb,ckt,timeStep);
        }
    }
    return(OK);
}
