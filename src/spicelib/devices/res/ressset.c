/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "resdefs.h"
#include "sperror.h"
#include "cktdefs.h"


int
RESsSetup(SENstruct *info, GENmodel *inModel)
        /* loop through all the devices and 
         * assign parameter #s to design parameters 
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
            here=here->RESnextInstance) {
	    
	    if (here->RESowner != ARCHme) continue;

            if(here->RESsenParmNo){
                here->RESsenParmNo = ++(info->SENparms);
            }
        }
    }
    return(OK);
}
