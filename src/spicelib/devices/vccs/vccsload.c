/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
VCCSload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the source models */
    for( ; model != NULL; model = VCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ;
                here=VCCSnextInstance(here)) {
            
            *(here->VCCSposContPosPtr) += here->VCCScoeff ;
            *(here->VCCSposContNegPtr) -= here->VCCScoeff ;
            *(here->VCCSnegContPosPtr) -= here->VCCScoeff ;
            *(here->VCCSnegContNegPtr) += here->VCCScoeff ;
        }
    }
    return(OK);
}
