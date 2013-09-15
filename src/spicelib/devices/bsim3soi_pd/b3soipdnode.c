/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"

int
B3SOIPDnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inModel ;
    B3SOIPDinstance *here ;

    /* loop through all the B3SOIPD models */
    for ( ; model != NULL ; model = model->B3SOIPDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIPDinstances ; here != NULL ; here = here->B3SOIPDnextInstance)
        {
            ckt->CKTnodeIsLinear [here->B3SOIPDdNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIPDsNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIPDbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIPDtempNode] = 0 ;
            if (here->B3SOIPDdebugMod != 0)
            {
                ckt->CKTnodeIsLinear [here->B3SOIPDvbsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDidsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDicNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDibsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDibdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDiiiNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDigNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDgiggNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDgigdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDgigbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDigidlNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDitunNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDibpNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDcbbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDcbdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDcbgNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDqbfNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDqjsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIPDqjdNode] = 0 ;
            }
            ckt->CKTnodeIsLinear [here->B3SOIPDgNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIPDeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIPDpNode] = 0 ;
        }
    }

    return (OK) ;
}
