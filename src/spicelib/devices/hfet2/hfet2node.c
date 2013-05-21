/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"

int
HFET2nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            ckt->CKTnodeIsLinear [here->HFET2sourcePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HFET2drainPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HFET2gateNode] = 0 ;
        }
    }

    return (OK) ;
}
