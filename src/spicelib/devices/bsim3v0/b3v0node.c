/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"

int
BSIM3v0nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM3v0dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v0sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v0qNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v0gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v0bNode] = 0 ;
        }
    }

    return (OK) ;
}
