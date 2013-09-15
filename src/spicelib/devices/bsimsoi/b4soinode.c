/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/sperror.h"

int
B4SOInodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel ;
    B4SOIinstance *here ;

    /* loop through all the BSIMSOI models */
    for ( ; model != NULL ; model = model->B4SOInextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances ; here != NULL ; here = here->B4SOInextInstance)
        {
            ckt->CKTnodeIsLinear [here->B4SOIdNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIsNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOItempNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIgNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIgNodeMid] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIdbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIsbNode] = 0 ;
            if (here->B4SOIdebugMod != 0)
            {
                ckt->CKTnodeIsLinear [here->B4SOIvbsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIidsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIicNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIibsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIibdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIiiiNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIigNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIgiggNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIgigdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIgigbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIigidlNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIitunNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIibpNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIcbbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIcbdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIcbgNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIqbfNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIqjsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B4SOIqjdNode] = 0 ;
            }
            ckt->CKTnodeIsLinear [here->B4SOIeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B4SOIpNode] = 0 ;
        }
    }

    return (OK) ;
}
