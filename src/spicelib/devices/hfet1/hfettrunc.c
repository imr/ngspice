
#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
HFETAtrunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    CKTcircuit *ckt;
    double *timeStep;
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;

    for( ; model != NULL; model = model->HFETAnextModel) {
        for(here=model->HFETAinstances;here!=NULL;here = here->HFETAnextInstance){
            CKTterr(here->HFETAqgs,ckt,timeStep);
            CKTterr(here->HFETAqgd,ckt,timeStep);
        }
    }
    return(OK);
}
