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
MOS6getic(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->MOS6nextModel) {
        for(here = model->MOS6instances; here ; here = here->MOS6nextInstance) {
	    if (here->MOS6owner != ARCHme) continue;
        
            if(!here->MOS6icVBSGiven) {
                here->MOS6icVBS = 
                        *(ckt->CKTrhs + here->MOS6bNode) - 
                        *(ckt->CKTrhs + here->MOS6sNode);
            }
            if(!here->MOS6icVDSGiven) {
                here->MOS6icVDS = 
                        *(ckt->CKTrhs + here->MOS6dNode) - 
                        *(ckt->CKTrhs + here->MOS6sNode);
            }
            if(!here->MOS6icVGSGiven) {
                here->MOS6icVGS = 
                        *(ckt->CKTrhs + here->MOS6gNode) - 
                        *(ckt->CKTrhs + here->MOS6sNode);
            }
        }
    }
    return(OK);
}
