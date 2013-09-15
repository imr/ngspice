/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"

int
ASRCnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel *)inModel ;
    ASRCinstance *here ;

    /* loop through all the ASRC models */
    for ( ; model != NULL ; model = model->ASRCnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->ASRCinstances ; here != NULL ; here = here->ASRCnextInstance)
        {
            ckt->CKTnodeIsLinear [here->ASRCposNode] = 0 ;
            ckt->CKTnodeIsLinear [here->ASRCnegNode] = 0 ;
        }
    }

    return (OK) ;
}
