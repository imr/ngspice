/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* load the voltage source structure with those pointers needed later 
 * for fast matrix loading 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
VCVSsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;
    int error;
    CKTnode *tmp;

    NG_IGNORE(states);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCVSinstances(model); here != NULL ;
                here=VCVSnextInstance(here)) {
            
            if(here->VCVSposNode == here->VCVSnegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted VCVS", here->VCVSname);
                return(E_UNSUPP);
            }

            if(here->VCVSbranch == 0) {
                error = CKTmkCur(ckt,&tmp,here->VCVSname,"branch");
                if(error) return(error);
                here->VCVSbranch = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(VCVSposIbrPtr, VCVSposNode, VCVSbranch);
            TSTALLOC(VCVSnegIbrPtr, VCVSnegNode, VCVSbranch);
            TSTALLOC(VCVSibrPosPtr, VCVSbranch, VCVSposNode);
            TSTALLOC(VCVSibrNegPtr, VCVSbranch, VCVSnegNode);
            TSTALLOC(VCVSibrContPosPtr, VCVSbranch, VCVScontPosNode);
            TSTALLOC(VCVSibrContNegPtr, VCVSbranch, VCVScontNegNode);
        }
    }
    return(OK);
}

int
VCVSunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model;
    VCVSinstance *here;

    for (model = (VCVSmodel *)inModel; model != NULL;
	    model = VCVSnextModel(model))
    {
        for (here = VCVSinstances(model); here != NULL;
                here=VCVSnextInstance(here))
	{
	    if (here->VCVSbranch > 0)
		CKTdltNNum(ckt, here->VCVSbranch);
            here->VCVSbranch = 0;
	}
    }
    return OK;
}
