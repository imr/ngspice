/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim2def.h"
#include "sperror.h"
#include "suffix.h"


int
B2getic(GENmodel *inModel, CKTcircuit *ckt)
{

    B2model *model = (B2model*)inModel;
    B2instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->B2nextModel) {
        for(here = model->B2instances; here ; here = here->B2nextInstance) {
	    if (here->B2owner != ARCHme) continue;
        
            if(!here->B2icVBSGiven) {
                here->B2icVBS = 
                        *(ckt->CKTrhs + here->B2bNode) - 
                        *(ckt->CKTrhs + here->B2sNode);
            }
            if(!here->B2icVDSGiven) {
                here->B2icVDS = 
                        *(ckt->CKTrhs + here->B2dNode) - 
                        *(ckt->CKTrhs + here->B2sNode);
            }
            if(!here->B2icVGSGiven) {
                here->B2icVGS = 
                        *(ckt->CKTrhs + here->B2gNode) - 
                        *(ckt->CKTrhs + here->B2sNode);
            }
        }
    }
    return(OK);
}

