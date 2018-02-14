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
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCCSsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;
    double  vc;
    double  ivc;


    /*  loop through all the source models */
    for( ; model != NULL; model = VCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ;
                here=VCCSnextInstance(here)) {

	    if (here->VCCSsenParmNo){
                vc = *(ckt->CKTrhsOld + here->VCCScontPosNode)
                        -   *(ckt->CKTrhsOld + here->VCCScontNegNode);
                ivc = *(ckt->CKTirhsOld + here->VCCScontPosNode)
                        -   *(ckt->CKTirhsOld + here->VCCScontNegNode);
                *(ckt->CKTsenInfo->SEN_RHS[here->VCCSposNode] + 
                        here->VCCSsenParmNo) -= vc;
                *(ckt->CKTsenInfo->SEN_iRHS[here->VCCSposNode] +
                        here->VCCSsenParmNo) -= ivc;
                *(ckt->CKTsenInfo->SEN_RHS[here->VCCSnegNode] +
                        here->VCCSsenParmNo) += vc;
                *(ckt->CKTsenInfo->SEN_iRHS[here->VCCSnegNode] +
                        here->VCCSsenParmNo) += ivc;
            }
        }
    }
    return(OK);
}


