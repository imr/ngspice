/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current ac sensitivity information
 * into the array previously provided 
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "ccvsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CCVSsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;
    double   ic,i_ic;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->CCVSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCVSinstances; here != NULL ;
                here=here->CCVSnextInstance) {
	    if (here->CCVSowner != ARCHme) continue;

            if(here->CCVSsenParmNo){
                ic = *(ckt->CKTrhsOld + here->CCVScontBranch);
                i_ic = *(ckt->CKTirhsOld + here->CCVScontBranch);

                *(ckt->CKTsenInfo->SEN_RHS[here->CCVSbranch] +
                        here->CCVSsenParmNo) -= ic;
                *(ckt->CKTsenInfo->SEN_iRHS[here->CCVSbranch] + 
                        here->CCVSsenParmNo) -= i_ic;
            }
        }
    }
    return(OK);
}


