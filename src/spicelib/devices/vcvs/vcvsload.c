/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "vcvsdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
VCVSload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->VCVSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VCVSinstances; here != NULL ;
                here=here->VCVSnextInstance) {
	    if (here->VCVSowner != ARCHme) continue;
            
            *(here->VCVSposIbrptr) += 1.0 ;
            *(here->VCVSnegIbrptr) -= 1.0 ;
            *(here->VCVSibrPosptr) += 1.0 ;
            *(here->VCVSibrNegptr) -= 1.0 ;
            *(here->VCVSibrContPosptr) -= here->VCVScoeff ;
            *(here->VCVSibrContNegptr) += here->VCVScoeff ;
        }
    }
    return(OK);
}
