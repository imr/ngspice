/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "ccvsdefs.h"
#include "suffix.h"


/*ARGSUSED*/
int
CCVSpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)

        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->CCVSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCVSinstances; here != NULL ;
                here=here->CCVSnextInstance) {
	    if (here->CCVSowner != ARCHme) continue;
            
            *(here->CCVSposIbrptr) += 1.0 ;
            *(here->CCVSnegIbrptr) -= 1.0 ;
            *(here->CCVSibrPosptr) += 1.0 ;
            *(here->CCVSibrNegptr) -= 1.0 ;
            *(here->CCVSibrContBrptr) += here->CCVScoeff ;
        }
    }
    return(OK);
}
