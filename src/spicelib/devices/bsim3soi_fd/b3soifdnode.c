/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"

int
B3SOIFDnodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel ;
    B3SOIFDinstance *here ;

    /* loop through all the B3SOIFD models */
    for ( ; model != NULL ; model = model->B3SOIFDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances ; here != NULL ; here = here->B3SOIFDnextInstance)
        {
            ckt->CKTnodeIsLinear [here->B3SOIFDdNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIFDsNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIFDbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIFDtempNode] = 0 ;
            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
                ckt->CKTnodeIsLinear [here->B3SOIFDvbsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDidsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDicNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDibsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDibdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDiiiNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDigidlNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDitunNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDibpNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDabeffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvbs0effNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvbseffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDxcNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDcbbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDcbdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDcbgNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqbNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqbfNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqjsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqjdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDgmNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDgmbsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDgdsNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDgmeNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvbs0teffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvthNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvgsteffNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDxcsatNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqaccNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqsub0Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqsubs1Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqsubs2Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqeNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqdNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDqgNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvdscvNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDvcscvNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDcbeNode] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDdum1Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDdum2Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDdum3Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDdum4Node] = 0 ;
                ckt->CKTnodeIsLinear [here->B3SOIFDdum5Node] = 0 ;
            }
            ckt->CKTnodeIsLinear [here->B3SOIFDgNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIFDeNode] = 0 ;
            ckt->CKTnodeIsLinear [here->B3SOIFDpNode] = 0 ;
        }
    }

    return (OK) ;
}
