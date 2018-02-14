/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"


int
RESsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current ac sensitivity info into the 
         * array previously provided 
	 */
	 
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;
    double value;
    double ivalue;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = RESnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ;
                here=RESnextInstance(here)) {

            if(here->RESsenParmNo){
                value = *(ckt->CKTrhsOld+here->RESposNode) -
                    *(ckt->CKTrhsOld+here->RESnegNode);
                ivalue = *(ckt->CKTirhsOld+here->RESposNode) -
                    *(ckt->CKTirhsOld+here->RESnegNode);
                value *= here->RESacConduct;
                value *= here->RESacConduct;
                ivalue *= here->RESacConduct;
                ivalue *= here->RESacConduct;

                /* load the RHS matrix */
                *(ckt->CKTsenInfo->SEN_RHS[here->RESposNode] + 
                        here->RESsenParmNo) += value;
                *(ckt->CKTsenInfo->SEN_iRHS[here->RESposNode] +
                        here->RESsenParmNo) += ivalue;
                *(ckt->CKTsenInfo->SEN_RHS[here->RESnegNode] + 
                        here->RESsenParmNo) -= value;
                *(ckt->CKTsenInfo->SEN_iRHS[here->RESnegNode] + 
                        here->RESsenParmNo) -= ivalue;
            }
        }
    }
    return(OK);
}
