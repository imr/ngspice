/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
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
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = HICUMnextModel(model)) {
        for(here = HICUMinstances(model); here ; here = HICUMnextInstance(here)) {

            if(!here->HICUMicVBGiven) {
                here->HICUMicVB =
                        *(ckt->CKTrhs + here->HICUMbaseNode);
            }
            if(!here->HICUMicVCGiven) {
                here->HICUMicVC =
                        *(ckt->CKTrhs + here->HICUMcollNode);
            }
            if(!here->HICUMicVEGiven) {
                here->HICUMicVE =
                        *(ckt->CKTrhs + here->HICUMemitNode);
            }
            if(!here->HICUMicVBiGiven) {
                here->HICUMicVBi =
                        *(ckt->CKTrhs + here->HICUMbaseBINode);
            }
            if(!here->HICUMicVBpGiven) {
                here->HICUMicVBp =
                        *(ckt->CKTrhs + here->HICUMbaseBPNode);
            }
            if(!here->HICUMicVCiGiven) {
                here->HICUMicVCi =
                        *(ckt->CKTrhs + here->HICUMcollCINode);
            }
            if(!here->HICUMicVtGiven) {
                here->HICUMicVt =
                        *(ckt->CKTrhs + here->HICUMtempNode);
            }
            if(!here->HICUMicVEiGiven) {
                here->HICUMicVEi =
                        *(ckt->CKTrhs + here->HICUMemitEINode);
            }
        }
    }
    return(OK);
}
