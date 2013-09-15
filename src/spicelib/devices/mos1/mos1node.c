/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"

int
MOS1nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = model->MOS1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS1instances ; here != NULL ; here = here->MOS1nextInstance)
        {
            ckt->CKTnodeIsLinear [here->MOS1dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS1sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS1gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS1bNode] = 0 ;
        }
    }

    return (OK) ;
}
