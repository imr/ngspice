/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "mos2defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS2getic(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->MOS2nextModel) {
        for(here = model->MOS2instances; here ; here = here->MOS2nextInstance) {
	    if (here->MOS2owner != ARCHme) continue;
        
            if(!here->MOS2icVBSGiven) {
                here->MOS2icVBS = 
                        *(ckt->CKTrhs + here->MOS2bNode) - 
                        *(ckt->CKTrhs + here->MOS2sNode);
            }
            if(!here->MOS2icVDSGiven) {
                here->MOS2icVDS = 
                        *(ckt->CKTrhs + here->MOS2dNode) - 
                        *(ckt->CKTrhs + here->MOS2sNode);
            }
            if(!here->MOS2icVGSGiven) {
                here->MOS2icVGS = 
                        *(ckt->CKTrhs + here->MOS2gNode) - 
                        *(ckt->CKTrhs + here->MOS2sNode);
            }
        }
    }
    return(OK);
}
