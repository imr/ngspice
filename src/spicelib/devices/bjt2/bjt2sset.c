/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjt2defs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


int
BJT2sSetup(SENstruct *info, GENmodel *inModel)
{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;

#ifdef STEPDEBUG
    printf(" BJT2sensetup \n");
#endif /* STEPDEBUG */

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->BJT2nextModel ) {


        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {
            if (here->BJT2owner != ARCHme) continue;

            if(here->BJT2senParmNo){
                here->BJT2senParmNo = ++(info->SENparms);
                here->BJT2senPertFlag = OFF;
            }
            if((here->BJT2sens = (double *)MALLOC(55*sizeof(double))) ==
                NULL) return(E_NOMEM);
        }
    }
    return(OK);
}

