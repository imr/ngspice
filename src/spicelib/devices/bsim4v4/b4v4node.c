/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"

int
BSIM4v4nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel ;
    BSIM4v4instance *here ;

    /* loop through all the BSIM4v4 models */
    for ( ; model != NULL ; model = model->BSIM4v4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances ; here != NULL ; here = here->BSIM4v4nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM4v4dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4gNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4gNodeMid] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4dbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4bNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4sbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4v4qNode] = 0 ;
        }
    }

    return (OK) ;
}
