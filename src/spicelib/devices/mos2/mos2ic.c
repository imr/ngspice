/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS2getic(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = MOS2nextModel(model)) {
        for(here = MOS2instances(model); here ; here = MOS2nextInstance(here)) {

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
