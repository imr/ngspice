/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current ac sensitivity information
 *  into the array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCCSsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;
    double ic;
    double i_ic;

    /*  loop through all the CCCS models */
    for( ; model != NULL; model = CCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCCSinstances(model); here != NULL ;
                here=CCCSnextInstance(here)) {

            if(here->CCCSsenParmNo){

                ic = *(ckt->CKTrhsOld + here->CCCScontBranch);
                i_ic = *(ckt->CKTirhsOld + here->CCCScontBranch);
                *(ckt->CKTsenInfo->SEN_RHS[here->CCCSposNode] + 
                        here->CCCSsenParmNo) -= ic;
                *(ckt->CKTsenInfo->SEN_iRHS[here->CCCSposNode] +
                        here->CCCSsenParmNo) -= i_ic;
                *(ckt->CKTsenInfo->SEN_RHS[here->CCCSnegNode] + 
                        here->CCCSsenParmNo) += ic;
                *(ckt->CKTsenInfo->SEN_iRHS[here->CCCSnegNode] + 
                        here->CCCSsenParmNo) += i_ic;
            }
        }
    }
    return(OK);
}
