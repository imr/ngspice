/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "mos1defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS1trunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    double *timeStep;
{
    register MOS1model *model = (MOS1model *)inModel;
    register MOS1instance *here;

    for( ; model != NULL; model = model->MOS1nextModel) {
        for(here=model->MOS1instances;here!=NULL;here = here->MOS1nextInstance){
	    if (here->MOS1owner != ARCHme) continue;
        
            CKTterr(here->MOS1qgs,ckt,timeStep);
            CKTterr(here->MOS1qgd,ckt,timeStep);
            CKTterr(here->MOS1qgb,ckt,timeStep);
        }
    }
    return(OK);
}
