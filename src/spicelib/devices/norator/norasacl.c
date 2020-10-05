/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
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
#include "noradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NORAsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    NORAmodel *model = (NORAmodel *)inModel;
    NORAinstance *here;
    double   vc;
    double   ivc;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NORAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NORAinstances(model); here != NULL ;
                here=NORAnextInstance(here)) {

	    if(here->NORAsenParmNo){

                vc = *(ckt->CKTrhsOld + here->NORAcontPosNode)
                        -  *(ckt->CKTrhsOld + here->NORAcontNegNode);
                ivc = *(ckt->CKTirhsOld + here->NORAcontPosNode)
                        -  *(ckt->CKTirhsOld + here->NORAcontNegNode);

                *(ckt->CKTsenInfo->SEN_RHS[here->NORAbranch] +
                        here->NORAsenParmNo) += vc;
                *(ckt->CKTsenInfo->SEN_iRHS[here->NORAbranch] +
                        here->NORAsenParmNo) += ivc;
            }
        }
    }
    return(OK);
}

