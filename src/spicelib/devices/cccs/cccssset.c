/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "cccsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CCCSsSetup(SENstruct *info, GENmodel *inModel)
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;

    /*  loop through all the CCCS models */
    for( ; model != NULL; model = model->CCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
                here=here->CCCSnextInstance) {
	    if (here->CCCSowner != ARCHme) continue;

            if(here->CCCSsenParmNo){
                here->CCCSsenParmNo = ++(info->SENparms);
            }

        }
    }
    return(OK);
}

