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
#include "vcvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCVSsSetup(SENstruct *info, GENmodel *inModel)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCVSinstances(model); here != NULL ;
                here=VCVSnextInstance(here)) {

            if(here->VCVSsenParmNo){
                here->VCVSsenParmNo = ++(info->SENparms);
            }

        }
    }
    return(OK);
}
