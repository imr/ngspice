/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

    /* loop through all the devices and 
     * allocate parameter #s to design parameters 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
INDsSetup(SENstruct *info, GENmodel *inModel)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

	    if(here->INDsenParmNo){
                here->INDsenParmNo = ++(info->SENparms);
            }
        }
    }
    return(OK);
}
