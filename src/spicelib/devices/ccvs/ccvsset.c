/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* load the voltage source structure with those pointers needed later 
 * for fast matrix loading 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
CCVSsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;
    int error;
    CKTnode *tmp;

    NG_IGNORE(states);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = CCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ;
                here=CCVSnextInstance(here)) {
            
            if(here->CCVSposNode == here->CCVSnegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted CCVS", here->CCVSname);
                return(E_UNSUPP);
            }

            if(here->CCVSbranch==0) {
                error = CKTmkCur(ckt,&tmp,here->CCVSname, "branch");
                if(error) return(error);
                here->CCVSbranch = tmp->number;
            }
            here->CCVScontBranch = CKTfndBranch(ckt,here->CCVScontName);
            if(here->CCVScontBranch == 0) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: unknown controlling source %s", here->CCVSname, here->CCVScontName);
                return(E_BADPARM);
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(CCVSposIbrPtr, CCVSposNode, CCVSbranch);
            TSTALLOC(CCVSnegIbrPtr, CCVSnegNode, CCVSbranch);
            TSTALLOC(CCVSibrNegPtr, CCVSbranch, CCVSnegNode);
            TSTALLOC(CCVSibrPosPtr, CCVSbranch, CCVSposNode);
            TSTALLOC(CCVSibrContBrPtr, CCVSbranch, CCVScontBranch);
        }
    }
    return(OK);
}

int
CCVSunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model;
    CCVSinstance *here;

    for (model = (CCVSmodel *)inModel; model != NULL;
	    model = CCVSnextModel(model))
    {
        for (here = CCVSinstances(model); here != NULL;
                here=CCVSnextInstance(here))
	{
	    if (here->CCVSbranch) {
		CKTdltNNum(ckt, here->CCVSbranch);
		here->CCVSbranch = 0;
	    }
	}
    }
    return OK;
}
