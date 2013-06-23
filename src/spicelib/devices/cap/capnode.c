/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"

int
CAPnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel *)inModel ;
    CAPinstance *here ;
    int error ;

    /* loop through all the CAP models */
    for ( ; model != NULL ; model = model->CAPnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CAPinstances ; here != NULL ; here = here->CAPnextInstance)
        {
            error = CKTmkCurKCL (ckt, here->CAPposNode, &(here->KCLcurrentPos)) ;
            error = CKTmkCurKCL (ckt, here->CAPnegNode, &(here->KCLcurrentNeg)) ;
        }
    }

    return (OK) ;
}
