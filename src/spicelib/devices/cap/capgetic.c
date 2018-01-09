/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
CAPgetic(GENmodel *inModel, CKTcircuit *ckt)

{

    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = CAPnextModel(model)) {
        for(here = CAPinstances(model); here ; here = CAPnextInstance(here)) {
                
            if(!here->CAPicGiven) {
                here->CAPinitCond = 
                        *(ckt->CKTrhs + here->CAPposNode) - 
                        *(ckt->CKTrhs + here->CAPnegNode);
            }
        }
    }
    return(OK);
}
