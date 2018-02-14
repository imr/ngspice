/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
VCVSload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCVSinstances(model); here != NULL ;
                here=VCVSnextInstance(here)) {
            
            *(here->VCVSposIbrPtr) += 1.0 ;
            *(here->VCVSnegIbrPtr) -= 1.0 ;
            *(here->VCVSibrPosPtr) += 1.0 ;
            *(here->VCVSibrNegPtr) -= 1.0 ;
            *(here->VCVSibrContPosPtr) -= here->VCVScoeff ;
            *(here->VCVSibrContNegPtr) += here->VCVScoeff ;
        }
    }
    return(OK);
}
