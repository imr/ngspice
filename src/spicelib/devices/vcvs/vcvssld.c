/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current sensitivity information into the 
 * array previously provided 
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "vcvsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCVSsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;
    double   vc;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->VCVSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VCVSinstances; here != NULL ;
                here=here->VCVSnextInstance) {
	    if (here->VCVSowner != ARCHme) continue;

            if(here->VCVSsenParmNo){

                vc = *(ckt->CKTrhsOld + here->VCVScontPosNode)
                        -  *(ckt->CKTrhsOld + here->VCVScontNegNode);

                *(ckt->CKTsenInfo->SEN_RHS[here->VCVSbranch] + 
                        here->VCVSsenParmNo) += vc;
            }
        }
    }
    return(OK);
}

