/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "hfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
HFETAtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;

    for( ; model != NULL; model = model->HFETAnextModel) {
        for(here=model->HFETAinstances;here!=NULL;here = here->HFETAnextInstance){
            if (here->HFETAowner != ARCHme) continue;

            CKTterr(here->HFETAqgs,ckt,timeStep);
            CKTterr(here->HFETAqgd,ckt,timeStep);
        }
    }
    return(OK);
}
