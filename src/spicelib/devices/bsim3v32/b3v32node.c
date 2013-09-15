/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"

int
BSIM3v32nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = model->BSIM3v32nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances ; here != NULL ; here = here->BSIM3v32nextInstance)
        {
            ckt->CKTnodeIsLinear [here->BSIM3v32dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v32sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v32qNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v32gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM3v32bNode] = 0 ;
        }
    }

    return (OK) ;
}
