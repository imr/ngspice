/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

    /* loop through all the devices and 
     * allocate parameter #s to design parameters 
     */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "mos3defs.h"
#include "sperror.h"
#include "suffix.h"

int
MOS3sSetup(SENstruct *info, GENmodel *inModel)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = model->MOS3nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->MOS3instances; here != NULL ;
                here=here->MOS3nextInstance) {
	    if (here->MOS3owner != ARCHme) continue;

            if(here->MOS3senParmNo){
                if((here->MOS3sens_l)&&(here->MOS3sens_w)){
                    here->MOS3senParmNo = ++(info->SENparms);
                    ++(info->SENparms);/* MOS has two design parameters */
                }
                else{
                    here->MOS3senParmNo = ++(info->SENparms);
                }
            }
            here->MOS3senPertFlag = OFF;
            if((here->MOS3sens = (double *)MALLOC(72*sizeof(double))) == NULL) {
                return(E_NOMEM);
            }

        }
    }
    return(OK);
}


