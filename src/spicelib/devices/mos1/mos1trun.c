/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS1trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;

    for( ; model != NULL; model = MOS1nextModel(model)) {
        for(here=MOS1instances(model);here!=NULL;here = MOS1nextInstance(here)){
        
            CKTterr(here->MOS1qgs,ckt,timeStep);
            CKTterr(here->MOS1qgd,ckt,timeStep);
            CKTterr(here->MOS1qgb,ckt,timeStep);
        }
    }
    return(OK);
}
