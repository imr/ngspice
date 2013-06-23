/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"

int
RESnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;
    int error ;

    /* loop through all the RES models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            error = CKTmkCurKCL (ckt, here->RESposNode, &(here->KCLcurrentPos)) ;
            error = CKTmkCurKCL (ckt, here->RESnegNode, &(here->KCLcurrentNeg)) ;
        }
    }

    return (OK) ;
}
