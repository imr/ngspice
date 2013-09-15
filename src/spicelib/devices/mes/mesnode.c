/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
#include "ngspice/sperror.h"

int
MESnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            ckt->CKTnodeIsLinear [here->MESsourcePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MESdrainPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MESgateNode] = 0 ;
        }
    }

    return (OK) ;
}
