/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"

int
BSIM3nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = model->BSIM3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3instances ; here != NULL ; here = here->BSIM3nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM3dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3qNode] = 0 ;
        }
    }

    return (OK) ;
}
