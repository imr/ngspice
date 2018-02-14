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
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCCSsSetup(SENstruct *info, GENmodel *inModel)
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;

    /*  loop through all the CCCS models */
    for( ; model != NULL; model = CCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCCSinstances(model); here != NULL ;
                here=CCCSnextInstance(here)) {

            if(here->CCCSsenParmNo){
                here->CCCSsenParmNo = ++(info->SENparms);
            }

        }
    }
    return(OK);
}
