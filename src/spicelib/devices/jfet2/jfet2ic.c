/**********
Based on jfetic.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "jfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int
JFET2getic(GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->JFET2nextModel) {
        for(here = model->JFET2instances; here ; here = here->JFET2nextInstance) {
            if (here->JFET2owner != ARCHme) continue;
            if(!here->JFET2icVDSGiven) {
                here->JFET2icVDS = 
                        *(ckt->CKTrhs + here->JFET2drainNode) - 
                        *(ckt->CKTrhs + here->JFET2sourceNode);
            }
            if(!here->JFET2icVGSGiven) {
                here->JFET2icVGS = 
                        *(ckt->CKTrhs + here->JFET2gateNode) - 
                        *(ckt->CKTrhs + here->JFET2sourceNode);
            }
        }
    }
    return(OK);
}
