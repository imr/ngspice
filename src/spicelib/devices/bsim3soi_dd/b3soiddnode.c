/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"

int
B3SOIDDnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inModel ;
    B3SOIDDinstance *here ;

    /* loop through all the B3SOIDD models */
    for ( ; model != NULL ; model = model->B3SOIDDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIDDinstances ; here != NULL ; here = here->B3SOIDDnextInstance)
        {
            ckt->CKTnodeIsLinear [here->B3SOIDDdNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIDDsNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIDDbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIDDtempNode] = 0 ;
            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
                ckt->CKTnodeIsLinear [here->B3SOIDDvbsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDidsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDicNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDibsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDibdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDiiiNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDigidlNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDitunNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDibpNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDabeffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvbs0effNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvbseffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDxcNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDcbbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDcbdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDcbgNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqbfNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqjsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqjdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDgmNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDgmbsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDgdsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDgmeNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvbs0teffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvthNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvgsteffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDxcsatNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqaccNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqsub0Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqsubs1Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqsubs2Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqeNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDqgNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvdscvNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDvcscvNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDcbeNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDdum1Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDdum2Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDdum3Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDdum4Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIDDdum5Node] = 0 ;
            }
            ckt->CKTnodeIsLinear [here->B3SOIDDgNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIDDeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIDDpNode] = 0 ;
        }
    }

    return (OK) ;
}
