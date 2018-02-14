/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
CCVSload(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = CCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ;
                here=CCVSnextInstance(here)) {
            
            *(here->CCVSposIbrPtr) += 1.0 ;
            *(here->CCVSnegIbrPtr) -= 1.0 ;
            *(here->CCVSibrPosPtr) += 1.0 ;
            *(here->CCVSibrNegPtr) -= 1.0 ;
            *(here->CCVSibrContBrPtr) -= here->CCVScoeff ;
        }
    }
    return(OK);
}
