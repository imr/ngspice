/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "resdefs.h"
#include "sperror.h"


int
RESsLoad(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance sensitivity value into 
         * the array previously provided. 
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;
    double vres;
    double value;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {
 
        if (here->RESowner != ARCHme) continue;

            if(here->RESsenParmNo){
                vres = *(ckt->CKTrhsOld+here->RESposNode) -
                    *(ckt->CKTrhsOld+here->RESnegNode);
                value = vres * here->RESconduct * here->RESconduct;
		value = value * here->RESm * here->RESm;

                /* load the RHS matrix */
                *(ckt->CKTsenInfo->SEN_RHS[here->RESposNode] + 
                        here->RESsenParmNo) += value;
                *(ckt->CKTsenInfo->SEN_RHS[here->RESnegNode] + 
                        here->RESsenParmNo) -= value;
            }

        }
    }
    return(OK);
}

