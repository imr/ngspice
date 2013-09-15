/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"

int
MOS9nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            ckt->CKTnodeIsLinear [here->MOS9dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS9sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS9gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS9bNode] = 0 ;
        }
    }

    return (OK) ;
}
