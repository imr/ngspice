/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"

int
HSMHVnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel ;
    HSMHVinstance *here ;

    /* loop through all the HSMHV models */
    for ( ; model != NULL ; model = model->HSMHVnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances ; here != NULL ; here = here->HSMHVnextInstance)
        {
            ckt->CKTnodeIsLinear [here->HSMHVdNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVsNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVgNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVdbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVbNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVsbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVtempNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVqiNode] = 0 ;
            ckt->CKTnodeIsLinear [here->HSMHVqbNode] = 0 ;
        }
    }

    return (OK) ;
}
