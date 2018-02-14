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
HFETAgetic(GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = HFETAnextModel(model)) {
        for(here = HFETAinstances(model); here ; here = HFETAnextInstance(here)) {

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
