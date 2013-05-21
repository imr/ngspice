/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"

int
HSM2nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            ckt->CKTnodeIsLinear [here->HSM2dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSM2sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSM2gNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSM2dbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HSM2bNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSM2sbNode] = 0 ;
        }
    }

    return (OK) ;
}
