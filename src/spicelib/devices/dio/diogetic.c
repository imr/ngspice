/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOgetic(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = DIOnextModel(model)) {
        for(here = DIOinstances(model); here ; here = DIOnextInstance(here)) {

            if(!here->DIOinitCondGiven) {
                here->DIOinitCond = 
                        *(ckt->CKTrhs + here->DIOposNode) - 
                        *(ckt->CKTrhs + here->DIOnegNode);
            }
        }
    }
    return(OK);
}
