/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPsSetup(SENstruct *info, GENmodel *inModel)

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = CAPnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CAPinstances(model); here != NULL ;
                here=CAPnextInstance(here)) {

            if(here->CAPsenParmNo){
                here->CAPsenParmNo = ++(info->SENparms);
            }
        }
    }
    return(OK);
}
