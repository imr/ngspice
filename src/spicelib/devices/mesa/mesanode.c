/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"

int
MESAnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = model->MESAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESAinstances ; here != NULL ; here = here->MESAnextInstance)
        {
            ckt->CKTnodeIsLinear [here->MESAsourcePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MESAdrainPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MESAgatePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MESAsourcePrmPrmNode] = 0 ;
            ckt->CKTnodeIsLinear [here->MESAdrainPrmPrmNode] = 0 ;
        }
    }

    return (OK) ;
}
