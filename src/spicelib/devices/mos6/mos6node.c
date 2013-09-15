/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"

int
MOS6nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            ckt->CKTnodeIsLinear [here->MOS6dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS6sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS6gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS6bNode] = 0 ;
        }
    }

    return (OK) ;
}
