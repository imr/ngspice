/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "resdefs.h"
#include "sperror.h"


int
RESload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {
            
            *(here->RESposPosptr) += here->RESconduct;
            *(here->RESnegNegptr) += here->RESconduct;
            *(here->RESposNegptr) -= here->RESconduct;
            *(here->RESnegPosptr) -= here->RESconduct;
        }
    }
    return(OK);
}

int
RESacload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {
            
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
