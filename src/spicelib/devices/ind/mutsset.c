/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice.h"
#include "ifsim.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"

#ifdef MUTUAL
/*ARGSUSED*/
int
MUTsSetup(SENstruct *info, GENmodel *inModel)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance *here;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->MUTnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->MUTinstances; here != NULL ;
                here=here->MUTnextInstance) {
	    if (here->MUTowner != ARCHme) continue;

            if(here->MUTsenParmNo){
                here->MUTsenParmNo = ++(info->SENparms);
            }


        }
    }
    return(OK);
}
#endif /* MUTUAL */

