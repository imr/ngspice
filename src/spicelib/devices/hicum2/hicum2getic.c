/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning, Markus Müller, Mario Krattenmacher
License: 3-clause BSD
**********/

/*
 * This routine gets the device initial conditions for the HICUMs
 * from the RHS vector
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HICUMgetic(GENmodel *inModel, CKTcircuit *ckt)
{

    HICUMmodel *model = (HICUMmodel*)inModel;
    HICUMinstance *here;
    /*
     * grab initial conditions out of rhs array.  User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = HICUMnextModel(model)) {
        for(here = HICUMinstances(model); here ; here = HICUMnextInstance(here)) {

            if(!here->HICUMicVBEGiven) {
                here->HICUMicVBE =
                        *(ckt->CKTrhs + here->HICUMbaseNode) - 
                        *(ckt->CKTrhs + here->HICUMemitNode);
            }
            if(!here->HICUMicVCEGiven) {
                here->HICUMicVCE =
                        *(ckt->CKTrhs + here->HICUMcollNode) - 
                        *(ckt->CKTrhs + here->HICUMemitNode);
            }
            if(!here->HICUMicVCSGiven) {
                here->HICUMicVCS =
                        *(ckt->CKTrhs + here->HICUMcollNode) - 
                        *(ckt->CKTrhs + here->HICUMsubsNode);
            }
        }
    }
    return(OK);
}
