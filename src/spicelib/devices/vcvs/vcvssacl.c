/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current ac sensitivity information into the 
 * array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCVSsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;
    double   vc;
    double   ivc;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VCVSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCVSinstances(model); here != NULL ;
                here=VCVSnextInstance(here)) {

	    if(here->VCVSsenParmNo){

                vc = *(ckt->CKTrhsOld + here->VCVScontPosNode)
                        -  *(ckt->CKTrhsOld + here->VCVScontNegNode);
                ivc = *(ckt->CKTirhsOld + here->VCVScontPosNode)
                        -  *(ckt->CKTirhsOld + here->VCVScontNegNode);

                *(ckt->CKTsenInfo->SEN_RHS[here->VCVSbranch] +
                        here->VCVSsenParmNo) += vc;
                *(ckt->CKTsenInfo->SEN_iRHS[here->VCVSbranch] +
                        here->VCVSsenParmNo) += ivc;
            }
        }
    }
    return(OK);
}

