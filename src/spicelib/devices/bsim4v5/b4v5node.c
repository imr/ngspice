/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"

int
BSIM4v5nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel ;
    BSIM4v5instance *here ;

    /* loop through all the BSIM4v5 models */
    for ( ; model != NULL ; model = model->BSIM4v5nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances ; here != NULL ; here = here->BSIM4v5nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM4v5dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5gNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5gNodeMid] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5dbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5bNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5sbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v5qNode] = 0 ;
        }
    }

    return (OK) ;
}
