/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VDMOSgetic(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = VDMOSnextModel(model)) {
        for(here = VDMOSinstances(model); here ; here = VDMOSnextInstance(here)) {
        
            if(!here->VDMOSicVDSGiven) {
                here->VDMOSicVDS = 
                        *(ckt->CKTrhs + here->VDMOSdNode) - 
                        *(ckt->CKTrhs + here->VDMOSsNode);
            }
            if(!here->VDMOSicVGSGiven) {
                here->VDMOSicVGS = 
                        *(ckt->CKTrhs + here->VDMOSgNode) - 
                        *(ckt->CKTrhs + here->VDMOSsNode);
            }
        }
    }
    return(OK);
}
