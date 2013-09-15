/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"

int
BSIM4v6nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel ;
    BSIM4v6instance *here ;

    /* loop through all the BSIM4v6 models */
    for ( ; model != NULL ; model = model->BSIM4v6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v6instances ; here != NULL ; here = here->BSIM4v6nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM4v6dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6gNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6gNodeMid] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6dbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6bNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6sbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v6qNode] = 0 ;
        }
    }

    return (OK) ;
}
