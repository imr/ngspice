/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* loop through all the devices and 
 * allocate parameter #s to design parameters 
 */

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "vccsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCCSsSetup(info,inModel)
    register SENstruct *info;
    GENmodel *inModel;
{
    register VCCSmodel *model = (VCCSmodel *)inModel;
    register VCCSinstance *here;

    /*  loop through all the current source models */
    for( ; model != NULL; model = model->VCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VCCSinstances; here != NULL ;
                here=here->VCCSnextInstance) {
	    if (here->VCCSowner != ARCHme) continue;

            if(here->VCCSsenParmNo){
                here->VCCSsenParmNo = ++(info->SENparms);
            }

        }
    }
    return(OK);
}

