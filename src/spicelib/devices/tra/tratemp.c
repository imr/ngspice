/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "tradefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
TRAtemp(GENmodel *inModel, CKTcircuit *ckt)
        /*
         * pre-process parameters for later use
         */
{
    TRAmodel *model = (TRAmodel *)inModel;
    TRAinstance *here;

    /*  loop through all the transmission line models */
    for( ; model != NULL; model = model->TRAnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->TRAinstances; here != NULL ;
                here=here->TRAnextInstance) {
	    if (here->TRAowner != ARCHme) continue;
            
            if(!here->TRAtdGiven) {
                here->TRAtd = here->TRAnl/here->TRAf;
            }
            here->TRAconduct = 1/here->TRAimped;
        }
    }
    return(OK);
}
