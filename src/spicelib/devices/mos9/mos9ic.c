/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS9getic(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = MOS9nextModel(model)) {
        for(here = MOS9instances(model); here ; here = MOS9nextInstance(here)) {

            if(!here->MOS9icVBSGiven) {
                here->MOS9icVBS = 
                        *(ckt->CKTrhs + here->MOS9bNode) - 
                        *(ckt->CKTrhs + here->MOS9sNode);
            }
            if(!here->MOS9icVDSGiven) {
                here->MOS9icVDS = 
                        *(ckt->CKTrhs + here->MOS9dNode) - 
                        *(ckt->CKTrhs + here->MOS9sNode);
            }
            if(!here->MOS9icVGSGiven) {
                here->MOS9icVGS = 
                        *(ckt->CKTrhs + here->MOS9gNode) - 
                        *(ckt->CKTrhs + here->MOS9sNode);
            }
        }
    }
    return(OK);
}
