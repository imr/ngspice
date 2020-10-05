/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
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
#include "nuladefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NULAsSetup(SENstruct *info, GENmodel *inModel)
{
    NULAmodel *model = (NULAmodel *)inModel;
    NULAinstance *here;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NULAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NULAinstances(model); here != NULL ;
                here=NULAnextInstance(here)) {

            if(here->NULAsenParmNo){
                here->NULAsenParmNo = ++(info->SENparms);
            }

        }
    }
    return(OK);
}
