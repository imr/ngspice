/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* actually load the current sensitivity information into the 
 * array  previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCCSsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;
    double  vc;

    /*  loop through all the source models */
    for( ; model != NULL; model = VCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ;
                here=VCCSnextInstance(here)) {

            if(here->VCCSsenParmNo){
                vc = *(ckt->CKTrhsOld + here->VCCScontPosNode)
                        -  *(ckt->CKTrhsOld + here->VCCScontNegNode);
                *(ckt->CKTsenInfo->SEN_RHS[here->VCCSposNode] + 
                        here->VCCSsenParmNo) -= vc;
                *(ckt->CKTsenInfo->SEN_RHS[here->VCCSnegNode] + 
                        here->VCCSsenParmNo) += vc;
            }
        }
    }
    return(OK);
}
