/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Created 2020 Florian Ballenegger - based on VCVS code.
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* load the voltage source structure with those pointers needed later 
 * for fast matrix loading 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "noradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
NORAsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    NORAmodel *model = (NORAmodel *)inModel;
    NORAinstance *here;
    int error;
    CKTnode *tmp;

    NG_IGNORE(states);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NORAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NORAinstances(model); here != NULL ;
                here=NORAnextInstance(here)) {
            
            if(here->NORAposNode == here->NORAnegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted NORATOR", here->NORAname);
                return(E_UNSUPP);
            }
	    	    
            if(here->NORAbranch == 0) {
                error = CKTmkCur(ckt,&tmp,here->NORAname,"branch");
                if(error) return(error);
                here->NORAbranch = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(NORAposIbrPtr, NORAposNode, NORAbranch);
            TSTALLOC(NORAnegIbrPtr, NORAnegNode, NORAbranch);
	    /*
            TSTALLOC(NORAibrContPosPtr, NORAbranch, NORAcontPosNode);
            TSTALLOC(NORAibrContNegPtr, NORAbranch, NORAcontNegNode);
	    */
        }
    }
    return(OK);
}

int
NORAunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    NORAmodel *model;
    NORAinstance *here;

    for (model = (NORAmodel *)inModel; model != NULL;
	    model = NORAnextModel(model))
    {
        for (here = NORAinstances(model); here != NULL;
                here=NORAnextInstance(here))
	{
	    if (here->NORAbranch > 0)
		CKTdltNNum(ckt, here->NORAbranch);
            here->NORAbranch = 0;
	}
    }
    return OK;
}
