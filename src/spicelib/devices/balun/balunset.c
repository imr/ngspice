/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "balundefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
BALUNsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    BALUNmodel *model = (BALUNmodel *)inModel;
    BALUNinstance *here;
    int error;
    CKTnode *tmp;

    NG_IGNORE(states);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = BALUNnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = BALUNinstances(model); here != NULL ;
                here=BALUNnextInstance(here)) {
            
            if(here->BALUNposNode == here->BALUNnegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted BALUN", here->BALUNname);
                return(E_UNSUPP);
            }
	    
	    if(here->BALUNcmNode == here->BALUNdiffNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted BALUN", here->BALUNname);
                return(E_UNSUPP);
            }
	    	    
            if(here->BALUNbranchpos == 0) {
                error = CKTmkCur(ckt,&tmp,here->BALUNname,"branchpos");
                if(error) return(error);
                here->BALUNbranchpos = tmp->number;
            }
	    if(here->BALUNbranchneg == 0) {
                error = CKTmkCur(ckt,&tmp,here->BALUNname,"branchneg");
                if(error) return(error);
                here->BALUNbranchneg = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

	    TSTALLOC(BALUNposIbrposPtr, BALUNposNode, BALUNbranchpos);
	    TSTALLOC(BALUNnegIbrnegPtr, BALUNnegNode, BALUNbranchneg);
	    TSTALLOC(BALUNcmIbrposPtr, BALUNcmNode, BALUNbranchpos);
	    TSTALLOC(BALUNcmIbrnegPtr, BALUNcmNode, BALUNbranchneg);
	    TSTALLOC(BALUNdiffIbrposPtr, BALUNdiffNode, BALUNbranchpos);
	    TSTALLOC(BALUNdiffIbrnegPtr, BALUNdiffNode, BALUNbranchneg);
	    	    
	    TSTALLOC(BALUNibrposDiffPtr, BALUNbranchpos, BALUNdiffNode);
	    TSTALLOC(BALUNibrposPosPtr, BALUNbranchpos, BALUNposNode);
	    TSTALLOC(BALUNibrposNegPtr, BALUNbranchpos, BALUNnegNode);
	    
	    TSTALLOC(BALUNibrnegCmPtr, BALUNbranchneg, BALUNcmNode);
	    TSTALLOC(BALUNibrnegPosPtr, BALUNbranchneg, BALUNposNode);
	    TSTALLOC(BALUNibrnegNegPtr, BALUNbranchneg, BALUNnegNode);
        }
    }
    return(OK);
}

int
BALUNunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    BALUNmodel *model;
    BALUNinstance *here;

    for (model = (BALUNmodel *)inModel; model != NULL;
	    model = BALUNnextModel(model))
    {
        for (here = BALUNinstances(model); here != NULL;
                here=BALUNnextInstance(here))
	{
	    if (here->BALUNbranchpos > 0)
		CKTdltNNum(ckt, here->BALUNbranchpos);
            here->BALUNbranchpos = 0;
	    if (here->BALUNbranchneg > 0)
		CKTdltNNum(ckt, here->BALUNbranchneg);
            here->BALUNbranchneg = 0;
	}
    }
    return OK;
}
