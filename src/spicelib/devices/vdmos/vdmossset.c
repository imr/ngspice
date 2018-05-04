/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSsSetup(SENstruct *info, GENmodel *inModel)
/* loop through all the devices and 
         * allocate parameter #s to design parameters 
         */
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
                here=VDMOSnextInstance(here)) {

            if(here->VDMOSsenParmNo){
                if((here->VDMOSsens_l)&&(here->VDMOSsens_w)){
                    here->VDMOSsenParmNo = ++(info->SENparms);
                    ++(info->SENparms);/* MOS has two design parameters */
                }
                else{
                    here->VDMOSsenParmNo = ++(info->SENparms);
                }
            }
            if((here->VDMOSsens = TMALLOC(double, 70)) == NULL) {
                return(E_NOMEM);
            }
            here->VDMOSsenPertFlag = OFF;

        }
    }
    return(OK);
}


