/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B1getic(GENmodel *inModel, CKTcircuit *ckt)
{

    B1model *model = (B1model*)inModel;
    B1instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = B1nextModel(model)) {
        for(here = B1instances(model); here ; here = B1nextInstance(here)) {
        
            if(!here->B1icVBSGiven) {
                here->B1icVBS = 
                        *(ckt->CKTrhs + here->B1bNode) - 
                        *(ckt->CKTrhs + here->B1sNode);
            }
            if(!here->B1icVDSGiven) {
                here->B1icVDS = 
                        *(ckt->CKTrhs + here->B1dNode) - 
                        *(ckt->CKTrhs + here->B1sNode);
            }
            if(!here->B1icVGSGiven) {
                here->B1icVGS = 
                        *(ckt->CKTrhs + here->B1gNode) - 
                        *(ckt->CKTrhs + here->B1sNode);
            }
        }
    }
    return(OK);
}
