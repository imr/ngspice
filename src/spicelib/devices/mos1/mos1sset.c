/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "mos1defs.h"
#include "sperror.h"
#include "suffix.h"

int
MOS1sSetup(info,inModel)
register SENstruct *info;
GENmodel *inModel;
/* loop through all the devices and 
         * allocate parameter #s to design parameters 
         */
{
    register MOS1model *model = (MOS1model *)inModel;
    register MOS1instance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = model->MOS1nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->MOS1instances; here != NULL ;
                here=here->MOS1nextInstance) {
	    if (here->MOS1owner != ARCHme) continue;

            if(here->MOS1senParmNo){
                if((here->MOS1sens_l)&&(here->MOS1sens_w)){
                    here->MOS1senParmNo = ++(info->SENparms);
                    ++(info->SENparms);/* MOS has two design parameters */
                }
                else{
                    here->MOS1senParmNo = ++(info->SENparms);
                }
            }
            if((here->MOS1sens = (double *)MALLOC(70*sizeof(double))) == NULL) {
                return(E_NOMEM);
            }
            here->MOS1senPertFlag = OFF;

        }
    }
    return(OK);
}


