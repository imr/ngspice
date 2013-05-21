/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"

int
JFETnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = model->JFETnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFETinstances ; here != NULL ; here = here->JFETnextInstance)
        {
            ckt->CKTnodeIsLinear [here->JFETsourcePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->JFETdrainPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->JFETgateNode] = 0 ;
        }
    }

    return (OK) ;
}
