/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "cccsdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
CCCSload(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->CCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
                here=here->CCCSnextInstance) {
	    if (here->CCCSowner != ARCHme) continue;
            
            *(here->CCCSposContBrptr) += here->CCCScoeff ;
            *(here->CCCSnegContBrptr) -= here->CCCScoeff ;
        }
    }
    return(OK);
}
