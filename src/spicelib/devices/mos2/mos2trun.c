/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "mos2defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;

    for( ; model != NULL; model = model->MOS2nextModel) {
        for(here=model->MOS2instances;here!=NULL;here = here->MOS2nextInstance){
	    if (here->MOS2owner != ARCHme) continue;
        
            CKTterr(here->MOS2qgs,ckt,timeStep);
            CKTterr(here->MOS2qgd,ckt,timeStep);
            CKTterr(here->MOS2qgb,ckt,timeStep);
        }
    }
    return(OK);
}
