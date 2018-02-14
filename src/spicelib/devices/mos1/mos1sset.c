/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1sSetup(SENstruct *info, GENmodel *inModel)
/* loop through all the devices and 
         * allocate parameter #s to design parameters 
         */
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = MOS1nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ;
                here=MOS1nextInstance(here)) {

            if(here->MOS1senParmNo){
                if((here->MOS1sens_l)&&(here->MOS1sens_w)){
                    here->MOS1senParmNo = ++(info->SENparms);
                    ++(info->SENparms);/* MOS has two design parameters */
                }
                else{
                    here->MOS1senParmNo = ++(info->SENparms);
                }
            }
            if((here->MOS1sens = TMALLOC(double, 70)) == NULL) {
                return(E_NOMEM);
            }
            here->MOS1senPertFlag = OFF;

        }
    }
    return(OK);
}


