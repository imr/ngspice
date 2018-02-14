/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    double vcap;
    double ivcap;
    double val;
    double ival;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = CAPnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CAPinstances(model); here != NULL ;
                    here=CAPnextInstance(here)) {

            if(here->CAPsenParmNo){
                vcap = *(ckt->CKTrhsOld+here->CAPposNode) - 
                    *(ckt->CKTrhsOld+here->CAPnegNode);
                ivcap = *(ckt->CKTirhsOld+here->CAPposNode) - 
                    *(ckt->CKTirhsOld+here->CAPnegNode);   

                val = ckt->CKTomega * ivcap;
                ival = ckt->CKTomega * vcap;

                /* load the RHS matrix */

                *(ckt->CKTsenInfo->SEN_RHS[here->CAPposNode] + 
                        here->CAPsenParmNo) += val;
                *(ckt->CKTsenInfo->SEN_iRHS[here->CAPposNode] +
                        here->CAPsenParmNo) -= ival;
                *(ckt->CKTsenInfo->SEN_RHS[here->CAPnegNode] + 
                        here->CAPsenParmNo) -= val;
                *(ckt->CKTsenInfo->SEN_iRHS[here->CAPnegNode] + 
                        here->CAPsenParmNo) += ival;
            }
        }
    }
    return(OK);
}

