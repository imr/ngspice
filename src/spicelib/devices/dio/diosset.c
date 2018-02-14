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

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOsSetup(SENstruct *info, GENmodel *inModel)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            if(here->DIOsenParmNo){
                here->DIOsenParmNo = ++(info->SENparms);
                here->DIOsenPertFlag = OFF;
            }
            if((here->DIOsens = TMALLOC(double, 7))
                    == NULL) return(E_NOMEM);

        }
    }
    return(OK);
}
