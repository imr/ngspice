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
#include "smpdefs.h"
#include "cktdefs.h"
#include "mos9defs.h"
#include "sperror.h"
#include "suffix.h"

int
MOS9sSetup(SENstruct *info, GENmodel *inModel)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = model->MOS9nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->MOS9instances; here != NULL ;
                here=here->MOS9nextInstance) {
            if (here->MOS9owner != ARCHme) continue;


            if(here->MOS9senParmNo){
                if((here->MOS9sens_l)&&(here->MOS9sens_w)){
                    here->MOS9senParmNo = ++(info->SENparms);
                    ++(info->SENparms);/* MOS has two design parameters */
                }
                else{
                    here->MOS9senParmNo = ++(info->SENparms);
                }
            }
            here->MOS9senPertFlag = OFF;
            if((here->MOS9sens = (double *)MALLOC(72*sizeof(double))) == NULL) {
                return(E_NOMEM);
            }

        }
    }
    return(OK);
}


