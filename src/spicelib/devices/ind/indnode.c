/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"

int
INDnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel *)inModel ;
    INDinstance *here ;
    int error ;

    /* loop through all the IND models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            error = CKTmkCurKCL (ckt, here->INDposNode, &(here->KCLcurrentPos)) ;
            error = CKTmkCurKCL (ckt, here->INDnegNode, &(here->KCLcurrentNeg)) ;
        }
    }

    return (OK) ;
}
