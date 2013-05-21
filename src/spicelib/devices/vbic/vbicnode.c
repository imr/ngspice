/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"

int
VBICnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel ;
    VBICinstance *here ;

    /* loop through all the VBIC models */
    for ( ; model != NULL ; model = model->VBICnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VBICinstances ; here != NULL ; here = here->VBICnextInstance)
        {
            ckt->CKTnodeIsLinear [here->VBICcollCXNode] = 0 ;
            ckt->CKTnodeIsLinear [here->VBICbaseBXNode] = 0 ;
            ckt->CKTnodeIsLinear [here->VBICemitEINode] = 0 ;
            ckt->CKTnodeIsLinear [here->VBICsubsSINode] = 0 ;
            ckt->CKTnodeIsLinear [here->VBICcollCINode] = 0 ;
            ckt->CKTnodeIsLinear [here->VBICbaseBPNode] = 0 ;
            ckt->CKTnodeIsLinear [here->VBICbaseBINode] = 0 ;
        }
    }

    return (OK) ;
}
