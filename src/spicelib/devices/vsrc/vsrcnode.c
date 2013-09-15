/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"

int
VSRCnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    VSRCinstance *here ;
    int error ;

    /* loop through all the VSRC models */
    for ( ; model != NULL ; model = model->VSRCnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VSRCinstances ; here != NULL ; here = here->VSRCnextInstance)
        {
            error = CKTmkCurKCL (ckt, here->VSRCposNode, &(here->KCLcurrentPos)) ;
            error = CKTmkCurKCL (ckt, here->VSRCnegNode, &(here->KCLcurrentNeg)) ;
        }
    }

    return (OK) ;
}
