/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS9trunc(
    GENmodel *inModel,
    CKTcircuit *ckt,
    double *timeStep)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;

    for( ; model != NULL; model = model->MOS9nextModel) {
        for(here=model->MOS9instances;here!=NULL;here = here->MOS9nextInstance){

            CKTterr(here->MOS9qgs,ckt,timeStep);
            CKTterr(here->MOS9qgd,ckt,timeStep);
            CKTterr(here->MOS9qgb,ckt,timeStep);
        }
    }
    return(OK);
}
