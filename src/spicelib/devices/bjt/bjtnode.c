/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"

int
BJTnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel ;
    BJTinstance *here ;

    /* loop through all the BJT models */
    for ( ; model != NULL ; model = model->BJTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BJTinstances ; here != NULL ; here = here->BJTnextInstance)
        {
            ckt->CKTnodeIsLinear [here->BJTcolPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BJTbasePrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BJTemitPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BJTsubstNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BJTsubstConNode] = 0 ;
        }
    }

    return (OK) ;
}
