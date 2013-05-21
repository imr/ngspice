/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"

int
DIOnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = model->DIOnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->DIOinstances ; here != NULL ; here = here->DIOnextInstance)
        {
            ckt->CKTnodeIsLinear [here->DIOposPrimeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->DIOnegNode] = 0 ;
        }
    }

    return (OK) ;
}
