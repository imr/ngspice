/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "capdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
CAPsetup(matrix,inModel,ckt,states)
    register SMPmatrix *matrix;
    GENmodel *inModel;
    CKTcircuit *ckt;
    int *states;
        /* load the capacitor structure with those pointers needed later 
         * for fast matrix loading 
         */

{
    register CAPmodel *model = (CAPmodel*)inModel;
    register CAPinstance *here;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = model->CAPnextModel ) {

        /*Default Value Processing for Model Parameters */
        if (!model->CAPcjGiven) {
            model->CAPcj = 0;
        }
        if (!model->CAPcjswGiven){
             model->CAPcjsw = 0;
        }
        if (!model->CAPdefWidthGiven) {
            model->CAPdefWidth = 10.e-6;
        }
        if (!model->CAPnarrowGiven) {
            model->CAPnarrow = 0;
        }

        /* loop through all the instances of the model */
        for (here = model->CAPinstances; here != NULL ;
                here=here->CAPnextInstance) {
	    if (here->CAPowner != ARCHme) goto matrixpointers;

            /* Default Value Processing for Capacitor Instance */
            if (!here->CAPlengthGiven) {
                here->CAPlength = 0;
            }

            here->CAPqcap = *states;
            *states += 2;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 2 * (ckt->CKTsenInfo->SENparms);
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

matrixpointers:
            TSTALLOC(CAPposPosptr,CAPposNode,CAPposNode)
            TSTALLOC(CAPnegNegptr,CAPnegNode,CAPnegNode)
            TSTALLOC(CAPposNegptr,CAPposNode,CAPnegNode)
            TSTALLOC(CAPnegPosptr,CAPnegNode,CAPposNode)
        }
    }
    return(OK);
}

