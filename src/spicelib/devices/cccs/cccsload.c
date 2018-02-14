/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
CCCSload(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = CCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCCSinstances(model); here != NULL ;
                here=CCCSnextInstance(here)) {

            *(here->CCCSposContBrPtr) += here->CCCScoeff ;
            *(here->CCCSnegContBrPtr) -= here->CCCScoeff ;
        }
    }
    return(OK);
}
