/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

    /* loop through all the devices and 
     * allocate parameter #s to design parameters 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS9sSetup(SENstruct *info, GENmodel *inModel)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;

    /*  loop through all the models */
    for( ; model != NULL; model = MOS9nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS9instances(model); here != NULL ;
                here=MOS9nextInstance(here)) {

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
            if((here->MOS9sens = TMALLOC(double, 72)) == NULL) {
                return(E_NOMEM);
            }

        }
    }
    return(OK);
}


