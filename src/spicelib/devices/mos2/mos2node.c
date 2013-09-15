/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"

int
MOS2nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            ckt->CKTnodeIsLinear [here->MOS2dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS2sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS2gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS2bNode] = 0 ;
        }
    }

    return (OK) ;
}
