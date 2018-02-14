/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine gets the device initial conditions for the VBICs
 * from the RHS vector
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VBICgetic(GENmodel *inModel, CKTcircuit *ckt)
{

    VBICmodel *model = (VBICmodel*)inModel;
    VBICinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = VBICnextModel(model)) {
        for(here = VBICinstances(model); here ; here = VBICnextInstance(here)) {

            if(!here->VBICicVBEGiven) {
                here->VBICicVBE = 
                        *(ckt->CKTrhs + here->VBICbaseNode) - 
                        *(ckt->CKTrhs + here->VBICemitNode);
            }
            if(!here->VBICicVCEGiven) {
                here->VBICicVCE = 
                        *(ckt->CKTrhs + here->VBICcollNode) - 
                        *(ckt->CKTrhs + here->VBICemitNode);
            }
        }
    }
    return(OK);
}
