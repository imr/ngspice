/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 Paolo Nenzi
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "resdefs.h"
#include "sperror.h"


int 
RESsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit*ckt, int *state)
        /* load the resistor structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;


    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {
            
/*
 * Paolo Nenzi 2003
 *  The following lines are needed if I will move the defaulting code
 *  from REStemp to RESsetup, as in other (more recent ?) spice devices
 *  	
 * if (here->RESowner != ARCHme)
 *		goto matrixpointers;
 *       
 * matrixpointers:
 * 
 * put here instance parameter defaulting.
 */
	    
/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(RESposPosptr, RESposNode, RESposNode);
            TSTALLOC(RESnegNegptr, RESnegNode, RESnegNode);
            TSTALLOC(RESposNegptr, RESposNode, RESnegNode);
            TSTALLOC(RESnegPosptr, RESnegNode, RESposNode);
        }
    }
    return(OK);
}
