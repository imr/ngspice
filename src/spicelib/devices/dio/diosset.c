/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


int
DIOsSetup(SENstruct *info, GENmodel *inModel)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
	    if (here->DIOowner != ARCHme) continue;

            if(here->DIOsenParmNo){
                here->DIOsenParmNo = ++(info->SENparms);
                here->DIOsenPertFlag = OFF;
            }
            if((here->DIOsens = (double *)MALLOC(7*sizeof(double)))
                    == NULL) return(E_NOMEM);

        }
    }
    return(OK);
}

