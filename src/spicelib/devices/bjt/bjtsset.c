/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjtdefs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


int
BJTsSetup(SENstruct *info, GENmodel *inModel)
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;

#ifdef STEPDEBUG
    printf(" BJTsensetup \n");
#endif /* STEPDEBUG */

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->BJTnextModel ) {


        /* loop through all the instances of the model */
        for (here = model->BJTinstances; here != NULL ;
                here=here->BJTnextInstance) {
	    if (here->BJTowner != ARCHme) continue;

            if(here->BJTsenParmNo){
                here->BJTsenParmNo = ++(info->SENparms);
                here->BJTsenPertFlag = OFF;
            }
            if((here->BJTsens = (double *)MALLOC(55*sizeof(double))) ==
                NULL) return(E_NOMEM);
        }
    }
    return(OK);
}

