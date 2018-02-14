/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current sensitivity information
 * into the array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCVSsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;
    double   ic;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = CCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ;
                here=CCVSnextInstance(here)) {
            
            if(here->CCVSsenParmNo){
                ic = *(ckt->CKTrhsOld + here->CCVScontBranch);
                *(ckt->CKTsenInfo->SEN_RHS[here->CCVSbranch] + 
                        here->CCVSsenParmNo) -= ic;
            }
        }
    }
    return(OK);
}
