/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* load the current source structure with those pointers needed later 
     * for fast matrix loading 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
VCCSsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;

    NG_IGNORE(states);
    NG_IGNORE(ckt);

    /*  loop through all the current source models */
    for( ; model != NULL; model = model->VCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VCCSinstances; here != NULL ;
                here=here->VCCSnextInstance) {
            
/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(VCCSposContPosptr, VCCSposNode, VCCScontPosNode)
            TSTALLOC(VCCSposContNegptr, VCCSposNode, VCCScontNegNode)
            TSTALLOC(VCCSnegContPosptr, VCCSnegNode, VCCScontPosNode)
            TSTALLOC(VCCSnegContNegptr, VCCSnegNode, VCCScontNegNode)
        }
    }
    return(OK);
}
