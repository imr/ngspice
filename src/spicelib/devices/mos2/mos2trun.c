/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;

    for( ; model != NULL; model = MOS2nextModel(model)) {
        for(here=MOS2instances(model);here!=NULL;here = MOS2nextInstance(here)){

            CKTterr(here->MOS2qgs,ckt,timeStep);
            CKTterr(here->MOS2qgd,ckt,timeStep);
            CKTterr(here->MOS2qgb,ckt,timeStep);
        }
    }
    return(OK);
}
