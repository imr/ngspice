/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "mos1defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS1getic(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->MOS1nextModel) {
        for(here = model->MOS1instances; here ; here = here->MOS1nextInstance) {
	    if (here->MOS1owner != ARCHme) continue;
        
            if(!here->MOS1icVBSGiven) {
                here->MOS1icVBS = 
                        *(ckt->CKTrhs + here->MOS1bNode) - 
                        *(ckt->CKTrhs + here->MOS1sNode);
            }
            if(!here->MOS1icVDSGiven) {
                here->MOS1icVDS = 
                        *(ckt->CKTrhs + here->MOS1dNode) - 
                        *(ckt->CKTrhs + here->MOS1sNode);
            }
            if(!here->MOS1icVGSGiven) {
                here->MOS1icVGS = 
                        *(ckt->CKTrhs + here->MOS1gNode) - 
                        *(ckt->CKTrhs + here->MOS1sNode);
            }
        }
    }
    return(OK);
}
