/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"

int
B1nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;

    /* loop through all the BSIM1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            ckt->CKTnodeIsLinear [here->B1dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B1sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B1gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B1bNode] = 0 ;
        }
    }

    return (OK) ;
}
