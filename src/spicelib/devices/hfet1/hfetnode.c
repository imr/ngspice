/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"

int
HFETAnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = model->HFETAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFETAinstances ; here != NULL ; here = here->HFETAnextInstance)
        {
            ckt->CKTnodeIsLinear [here->HFETAsourcePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HFETAdrainPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HFETAgatePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HFETAdrainPrmPrmNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HFETAsourcePrmPrmNode] = 0 ;
        }
    }

    return (OK) ;
}
