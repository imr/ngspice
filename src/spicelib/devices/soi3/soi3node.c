/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"

int
SOI3nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel ;
    SOI3instance *here ;

    /* loop through all the SOI3 models */
    for ( ; model != NULL ; model = model->SOI3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SOI3instances ; here != NULL ; here = here->SOI3nextInstance)
        {
            ckt->CKTnodeIsLinear [here->SOI3dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3sNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3tout1Node] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3tout2Node] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3tout3Node] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3tout4Node] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3gfNode] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3gbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3bNode] = 0 ;
            ckt->CKTnodeIsLinear [here->SOI3toutNode] = 0 ;
        }
    }

    return (OK) ;
}
