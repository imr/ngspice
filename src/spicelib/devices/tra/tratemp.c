/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "tradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
TRAtemp(GENmodel *inModel, CKTcircuit *ckt)
        /*
         * pre-process parameters for later use
         */
{
    TRAmodel *model = (TRAmodel *)inModel;
    TRAinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the transmission line models */
    for( ; model != NULL; model = TRAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ;
                here=TRAnextInstance(here)) {
            
            if(!here->TRAtdGiven) {
                here->TRAtd = here->TRAnl/here->TRAf;
            }
            here->TRAconduct = 1/here->TRAimped;
        }
    }
    return(OK);
}
