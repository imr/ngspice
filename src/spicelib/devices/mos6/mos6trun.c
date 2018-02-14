/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS6trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance *here;

    for( ; model != NULL; model = MOS6nextModel(model)) {
        for(here=MOS6instances(model);here!=NULL;here = MOS6nextInstance(here)){
        
            CKTterr(here->MOS6qgs,ckt,timeStep);
            CKTterr(here->MOS6qgd,ckt,timeStep);
            CKTterr(here->MOS6qgb,ckt,timeStep);
        }
    }
    return(OK);
}
