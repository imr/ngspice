/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "bjtdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


int
BJTsSetup(SENstruct *info, GENmodel *inModel)
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;

#ifdef STEPDEBUG
    printf(" BJTsensetup \n");
#endif /* STEPDEBUG */

    /*  loop through all the diode models */
    for( ; model != NULL; model = BJTnextModel(model)) {


        /* loop through all the instances of the model */
        for (here = BJTinstances(model); here != NULL ;
                here=BJTnextInstance(here)) {

            if(here->BJTsenParmNo){
                here->BJTsenParmNo = ++(info->SENparms);
                here->BJTsenPertFlag = OFF;
            }
            if((here->BJTsens = TMALLOC(double, 55)) ==
                NULL) return(E_NOMEM);
        }
    }
    return(OK);
}
