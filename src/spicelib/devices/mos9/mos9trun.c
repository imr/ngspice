/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "mos9defs.h"
#include "sperror.h"
#include "suffix.h"

int
MOS9trunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    CKTcircuit *ckt;
    double *timeStep;
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;

    for( ; model != NULL; model = model->MOS9nextModel) {
        for(here=model->MOS9instances;here!=NULL;here = here->MOS9nextInstance){
            if (here->MOS9owner != ARCHme) continue;

            CKTterr(here->MOS9qgs,ckt,timeStep);
            CKTterr(here->MOS9qgd,ckt,timeStep);
            CKTterr(here->MOS9qgb,ckt,timeStep);
        }
    }
    return(OK);
}
