
#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
HFETAgetic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->HFETAnextModel) {
        for(here = model->HFETAinstances; here ; here = here->HFETAnextInstance) {
            if(!here->HFETAicVDSGiven) {
                here->HFETAicVDS = 
                        *(ckt->CKTrhs + here->HFETAdrainNode) - 
                        *(ckt->CKTrhs + here->HFETAsourceNode);
            }
            if(!here->HFETAicVGSGiven) {
                here->HFETAicVGS = 
                        *(ckt->CKTrhs + here->HFETAgateNode) - 
                        *(ckt->CKTrhs + here->HFETAsourceNode);
            }
        }
    }
    return(OK);
}
