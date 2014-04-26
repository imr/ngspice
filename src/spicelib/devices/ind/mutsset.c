/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
MUTsSetup(SENstruct *info, GENmodel *inModel)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance *here;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = MUTnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MUTinstances(model); here != NULL ;
                here=MUTnextInstance(here)) {

            if(here->MUTsenParmNo){
                here->MUTsenParmNo = ++(info->SENparms);
            }


        }
    }
    return(OK);
}
