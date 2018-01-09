/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine gets the device initial conditions for the BJTs
 * from the RHS vector
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BJTgetic(GENmodel *inModel, CKTcircuit *ckt)

{

    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = BJTnextModel(model)) {
        for(here = BJTinstances(model); here ; here = BJTnextInstance(here)) {

            if(!here->BJTicVBEGiven) {
                here->BJTicVBE = 
                        *(ckt->CKTrhs + here->BJTbaseNode) - 
                        *(ckt->CKTrhs + here->BJTemitNode);
            }
            if(!here->BJTicVCEGiven) {
                here->BJTicVCE = 
                        *(ckt->CKTrhs + here->BJTcolNode) - 
                        *(ckt->CKTrhs + here->BJTemitNode);
            }
        }
    }
    return(OK);
}
