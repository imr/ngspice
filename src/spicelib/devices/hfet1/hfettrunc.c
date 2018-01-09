/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HFETAtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;

    for( ; model != NULL; model = HFETAnextModel(model)) {
        for(here=HFETAinstances(model);here!=NULL;here = HFETAnextInstance(here)){

            CKTterr(here->HFETAqgs,ckt,timeStep);
            CKTterr(here->HFETAqgd,ckt,timeStep);
        }
    }
    return(OK);
}
