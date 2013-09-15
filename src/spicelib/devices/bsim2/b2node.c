/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/sperror.h"

int
B2nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;

    /* loop through all the BSIM2 models */
    for ( ; model != NULL ; model = model->B2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B2instances ; here != NULL ; here = here->B2nextInstance)
        {
            ckt->CKTnodeIsLinear [here->B2dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B2sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B2gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B2bNode] = 0 ;
        }
    }

    return (OK) ;
}
