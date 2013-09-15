/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"

int
MOS3nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = model->MOS3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS3instances ; here != NULL ; here = here->MOS3nextInstance)
        {
            ckt->CKTnodeIsLinear [here->MOS3dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS3sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS3gNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MOS3bNode] = 0 ;
        }
    }

    return (OK) ;
}
