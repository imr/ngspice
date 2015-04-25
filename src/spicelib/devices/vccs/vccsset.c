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
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            if (here->VCCSbranch == 0) {
                here->VCCSposPrimeNode = here->VCCSposNode;
            } else {
                here->VCCSposPrimeNode = here->VCCSbranch;
                TSTALLOC(VCCS_pos_ibr, VCCSposNode, VCCSbranch);
                TSTALLOC(VCCS_posPrime_ibr, VCCSposPrimeNode, VCCSbranch);
            }

            TSTALLOC(VCCSposPrimeContPosptr, VCCSposPrimeNode, VCCScontPosNode);
            TSTALLOC(VCCSposPrimeContNegptr, VCCSposPrimeNode, VCCScontNegNode);
            TSTALLOC(VCCSnegContPosptr, VCCSnegNode, VCCScontPosNode);
            TSTALLOC(VCCSnegContNegptr, VCCSnegNode, VCCScontNegNode);
        }
    }
    return(OK);
}


int
VCCSunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *) inModel;
    VCCSinstance *here;

    for (; model; model = model->VCCSnextModel)
        for (here = model->VCCSinstances; here; here = here->VCCSnextInstance)
            if (here->VCCSbranch) {
                CKTdltNNum(ckt, here->VCCSbranch);
                here->VCCSbranch = 0;
                here->VCCSposPrimeNode = 0;
            }

    return OK;
}
