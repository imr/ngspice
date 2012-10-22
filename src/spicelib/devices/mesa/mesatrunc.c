/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESAtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MESAmodel *model = (MESAmodel*)inModel;
    MESAinstance *here;

    for( ; model != NULL; model = model->MESAnextModel) {
        for(here=model->MESAinstances;here!=NULL;
            here = here->MESAnextInstance){

            CKTterr(here->MESAqgs,ckt,timeStep);
            CKTterr(here->MESAqgd,ckt,timeStep);
        }
    }
    return(OK);
}
