/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "mos6defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS6trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance *here;

    for( ; model != NULL; model = model->MOS6nextModel) {
        for(here=model->MOS6instances;here!=NULL;here = here->MOS6nextInstance){
	    if (here->MOS6owner != ARCHme) continue;
        
            CKTterr(here->MOS6qgs,ckt,timeStep);
            CKTterr(here->MOS6qgd,ckt,timeStep);
            CKTterr(here->MOS6qgb,ckt,timeStep);
        }
    }
    return(OK);
}
