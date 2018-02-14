/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS2sSetup(SENstruct *info, GENmodel *inModel)
/* loop through all the devices and 
         * allocate parameter #s to design parameters 
         */
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = MOS2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS2instances(model); here != NULL ;
                here=MOS2nextInstance(here)) {

            if(here->MOS2senParmNo){
                if((here->MOS2sens_l)&&(here->MOS2sens_w)){
                    here->MOS2senParmNo = ++(info->SENparms);
                    ++(info->SENparms);/* MOS has two design parameters */
                }
                else{
                    here->MOS2senParmNo = ++(info->SENparms);
                }
            }
            here->MOS2senPertFlag = OFF;
            if((here->MOS2sens = TMALLOC(double, 70)) == NULL) {
                return(E_NOMEM);
            }

        }
    }
    return(OK);
}


