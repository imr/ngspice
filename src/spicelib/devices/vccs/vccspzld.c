/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "vccsdefs.h"
#include "ngspice/suffix.h"
#include "vccsext.h"


/*ARGSUSED*/
int
VCCSpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    /*  loop through all the source models */
    for( ; model != NULL; model = model->VCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VCCSinstances; here != NULL ;
                here=here->VCCSnextInstance) {
            
            *(here->VCCSposContPosptr) += here->VCCScoeff ;
            *(here->VCCSposContNegptr) -= here->VCCScoeff ;
            *(here->VCCSnegContPosptr) -= here->VCCScoeff ;
            *(here->VCCSnegContNegptr) += here->VCCScoeff ;
        }
    }
    return(OK);
}
