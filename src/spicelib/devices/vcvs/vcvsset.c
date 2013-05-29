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
    for( ; model != NULL; model = model->VCVSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VCVSinstances; here != NULL ;
                here=here->VCVSnextInstance) {
            
            if(here->VCVSposNode == here->VCVSnegNode) {
                SPfrontEnd->IFerror (ERR_FATAL,
                        "instance %s is a shorted VCVS", &here->VCVSname);
                return(E_UNSUPP);
            }

            if(here->VCVSbranch == 0) {
                error = CKTmkCur(ckt,&tmp,here->VCVSname,"branch");
                if(error) return(error);
                here->VCVSbranch = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(VCVSposIbrptr, VCVSposNode, VCVSbranch)
            TSTALLOC(VCVSnegIbrptr, VCVSnegNode, VCVSbranch)
            TSTALLOC(VCVSibrPosptr, VCVSbranch, VCVSposNode)
            TSTALLOC(VCVSibrNegptr, VCVSbranch, VCVSnegNode)
            TSTALLOC(VCVSibrContPosptr, VCVSbranch, VCVScontPosNode)
            TSTALLOC(VCVSibrContNegptr, VCVSbranch, VCVScontNegNode)
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
	    model = model->VCVSnextModel)
    {
        for (here = model->VCVSinstances; here != NULL;
                here=here->VCVSnextInstance)
	{
	    if (here->VCVSbranch) {
		CKTdltNNum(ckt, here->VCVSbranch);
		here->VCVSbranch = 0;
	    }
	}
    }
    return OK;
}
