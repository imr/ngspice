/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCVSsSetup(SENstruct *info, GENmodel *inModel)
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = CCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ;
                here=CCVSnextInstance(here)) {

            if(here->CCVSsenParmNo){
                here->CCVSsenParmNo = ++(info->SENparms);
            }
        }
    }
    return(OK);
}
