/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "resdefs.h"
#include "sperror.h"


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
	RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
	     here = here->RESnextInstance) {
            
            *(here->RESposPosptr) += here->RESconduct;
            *(here->RESnegNegptr) += here->RESconduct;
            *(here->RESposNegptr) -= here->RESconduct;
            *(here->RESnegPosptr) -= here->RESconduct;
        }
    }
    return(OK);
}


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESacload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
	RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
	     here = here->RESnextInstance) {
            
            if(here->RESacresGiven) {
                *(here->RESposPosptr) += here->RESacConduct;
                *(here->RESnegNegptr) += here->RESacConduct;
                *(here->RESposNegptr) -= here->RESacConduct;
                *(here->RESnegPosptr) -= here->RESacConduct;
            } else {
                *(here->RESposPosptr) += here->RESconduct;
                *(here->RESnegNegptr) += here->RESconduct;
                *(here->RESposNegptr) -= here->RESconduct;
                *(here->RESnegPosptr) -= here->RESconduct;
            }
        }
    }
    return(OK);
}
