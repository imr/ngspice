/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"

int
INDsetup(matrix,inModel,ckt,states)
    register SMPmatrix *matrix;
    GENmodel *inModel;
    CKTcircuit *ckt;
    int *states;
        /* load the inductor structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    register INDmodel *model = (INDmodel*)inModel;
    register INDinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->INDnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->INDinstances; here != NULL ;
                here=here->INDnextInstance) {
	    if (here->INDowner != ARCHme) goto matrixpointers;

            here->INDflux = *states;
            *states += 2 ;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 2 * (ckt->CKTsenInfo->SENparms);
            }
            
matrixpointers:
            if(here->INDbrEq == 0) {
                error = CKTmkCur(ckt,&tmp,here->INDname,"branch");
                if(error) return(error);
                here->INDbrEq = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(INDposIbrptr,INDposNode,INDbrEq)
            TSTALLOC(INDnegIbrptr,INDnegNode,INDbrEq)
            TSTALLOC(INDibrNegptr,INDbrEq,INDnegNode)
            TSTALLOC(INDibrPosptr,INDbrEq,INDposNode)
            TSTALLOC(INDibrIbrptr,INDbrEq,INDbrEq)
        }
    }
    return(OK);
}

int
INDunsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
#ifndef HAS_BATCHSIM
    INDmodel *model;
    INDinstance *here;

    for (model = (INDmodel *)inModel; model != NULL;
	    model = model->INDnextModel)
    {
        for (here = model->INDinstances; here != NULL;
                here=here->INDnextInstance)
	{
	    if (here->INDbrEq) {
		CKTdltNNum(ckt, here->INDbrEq);
		here->INDbrEq = 0;
	    }
	}
    }
#endif
    return OK;
}
