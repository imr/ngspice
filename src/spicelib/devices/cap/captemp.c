/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* load the capacitor structure with those pointers needed later 
     * for fast matrix loading 
     */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "capdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
CAPtemp(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = model->CAPnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CAPinstances; here != NULL ;
                here=here->CAPnextInstance) {
	    if (here->CAPowner != ARCHme) continue;

            /* Default Value Processing for Capacitor Instance */
            if (!here->CAPwidthGiven) {
                here->CAPwidth = model->CAPdefWidth;
            }
            if (!here->CAPcapGiven)  {
                here->CAPcapac = 
                        model->CAPcj * 
                            (here->CAPwidth - model->CAPnarrow) * 
                            (here->CAPlength - model->CAPnarrow) + 
                        model->CAPcjsw * 2 * (
                            (here->CAPlength - model->CAPnarrow) +
                            (here->CAPwidth - model->CAPnarrow) );
            }
        }
    }
    return(OK);
}

