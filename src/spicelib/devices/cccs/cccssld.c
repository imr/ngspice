/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current sensitivity information
 *  into the array previously provided 
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "cccsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CCCSsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;
    double ic ;

    /*  loop through all the CCCS models */
    for( ; model != NULL; model = model->CCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
                here=here->CCCSnextInstance) {
            if(here->CCCSsenParmNo){

                ic = *(ckt->CKTrhsOld + here->CCCScontBranch);
                *(ckt->CKTsenInfo->SEN_RHS[here->CCCSposNode] + 
                        here->CCCSsenParmNo) -= ic;
                *(ckt->CKTsenInfo->SEN_RHS[here->CCCSnegNode] + 
                        here->CCCSsenParmNo) += ic;
            }
        }
    }
    return(OK);
}

