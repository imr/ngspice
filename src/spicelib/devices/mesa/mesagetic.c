/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "mesadefs.h"
#include "sperror.h"
#include "suffix.h"


int
MESAgetic(GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel*)inModel;
    MESAinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->MESAnextModel) {
        for(here = model->MESAinstances; here ; here = here->MESAnextInstance) {
            if (here->MESAowner != ARCHme) continue;

            if(!here->MESAicVDSGiven) {
                here->MESAicVDS = 
                        *(ckt->CKTrhs + here->MESAdrainNode) - 
                        *(ckt->CKTrhs + here->MESAsourceNode);
            }
            if(!here->MESAicVGSGiven) {
                here->MESAicVGS = 
                        *(ckt->CKTrhs + here->MESAgateNode) - 
                        *(ckt->CKTrhs + here->MESAsourceNode);
            }
        }
    }
    return(OK);
}
