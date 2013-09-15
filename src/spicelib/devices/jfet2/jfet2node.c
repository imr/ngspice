/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/sperror.h"

int
JFET2nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            ckt->CKTnodeIsLinear [here->JFET2sourcePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->JFET2drainPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->JFET2gateNode] = 0 ;
        }
    }

    return (OK) ;
}
