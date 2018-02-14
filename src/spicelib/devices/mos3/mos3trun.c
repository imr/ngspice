/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS3trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;

    for( ; model != NULL; model = MOS3nextModel(model)) {
        for(here=MOS3instances(model);here!=NULL;here = MOS3nextInstance(here)){
        
            CKTterr(here->MOS3qgs,ckt,timeStep);
            CKTterr(here->MOS3qgd,ckt,timeStep);
            CKTterr(here->MOS3qgb,ckt,timeStep);
        }
    }
    return(OK);
}
