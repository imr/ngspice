/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

/*
 * This routine gets the device initial conditions for the BJT2s
 * from the RHS vector
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


int
BJT2getic(GENmodel *inModel, CKTcircuit *ckt)
{

    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = model->BJT2nextModel) {
        for(here = model->BJT2instances; here ; here = here->BJT2nextInstance) {
            if (here->BJT2owner != ARCHme) continue;  
	    
	    if(!here->BJT2icVBEGiven) {
                here->BJT2icVBE = 
                        *(ckt->CKTrhs + here->BJT2baseNode) - 
                        *(ckt->CKTrhs + here->BJT2emitNode);
            }
            if(!here->BJT2icVCEGiven) {
                here->BJT2icVCE = 
                        *(ckt->CKTrhs + here->BJT2colNode) - 
                        *(ckt->CKTrhs + here->BJT2emitNode);
            }
        }
    }
    return(OK);
}
