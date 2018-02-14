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
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCCSsSetup(SENstruct *info, GENmodel *inModel)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;

    /*  loop through all the current source models */
    for( ; model != NULL; model = VCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ;
                here=VCCSnextInstance(here)) {

            if(here->VCCSsenParmNo){
                here->VCCSsenParmNo = ++(info->SENparms);
            }

        }
    }
    return(OK);
}
