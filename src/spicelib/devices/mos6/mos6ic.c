/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS6getic(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = MOS6nextModel(model)) {
        for(here = MOS6instances(model); here ; here = MOS6nextInstance(here)) {
        
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
