/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"

int
BSIM3v1nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM3v1dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v1sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v1qNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v1gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v1bNode] = 0 ;
        }
    }

    return (OK) ;
}
