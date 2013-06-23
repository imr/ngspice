/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

int
BSIM4nodeIsNonLinear (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    int error ;

    /* loop through all the BSIM4 models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {

#ifdef STEPDEBUG
            fprintf (stderr, "here->BSIM4dNodePrime: %d\n", here->BSIM4dNodePrime) ;
            fprintf (stderr, "here->BSIM4sNodePrime: %d\n", here->BSIM4sNodePrime) ;
            fprintf (stderr, "here->BSIM4gNodePrime: %d\n", here->BSIM4gNodePrime) ;
            fprintf (stderr, "here->BSIM4gNodeMid: %d\n", here->BSIM4gNodeMid) ;
            fprintf (stderr, "here->BSIM4dbNode: %d\n", here->BSIM4dbNode) ;
            fprintf (stderr, "here->BSIM4bNodePrime: %d\n", here->BSIM4bNodePrime) ;
            fprintf (stderr, "here->BSIM4sbNode: %d\n", here->BSIM4sbNode) ;
            fprintf (stderr, "here->BSIM4qNode: %d\n", here->BSIM4qNode) ;
#endif

            ckt->CKTnodeIsLinear [here->BSIM4dNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4sNodePrime] = 0 ;
//            ckt->CKTnodeIsLinear [here->BSIM4gNodePrime] = 0 ;
//            ckt->CKTnodeIsLinear [here->BSIM4gNodeMid] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4dbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4bNodePrime] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4sbNode] = 0 ;
            ckt->CKTnodeIsLinear [here->BSIM4qNode] = 0 ;


            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrime_1)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrime_2)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrime_3)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrime_4)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrime_5)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4gNodePrime, &(here->KCLcurrentgNodePrime_1)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4gNodePrime, &(here->KCLcurrentgNodePrime_2)) ;

            if (here->BSIM4rgateMod == 3)
                error = CKTmkCurKCL (ckt, here->BSIM4gNodeMid, &(here->KCLcurrentgNodeMid)) ;

            if (!here->BSIM4rbodyMod)
            {
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_5)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_6)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_5)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_6)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_7)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_8)) ;
            } else {
                error = CKTmkCurKCL (ckt, here->BSIM4dbNode, &(here->KCLcurrentdbNode_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4dbNode, &(here->KCLcurrentdbNode_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrime_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sbNode, &(here->KCLcurrentsbNode_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sbNode, &(here->KCLcurrentsbNode_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_5)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_6)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_7)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_8)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_9)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime_10)) ;
            }

            if (here->BSIM4trnqsMod)
                error = CKTmkCurKCL (ckt, here->BSIM4qNode, &(here->KCLcurrentqNode_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4qNode, &(here->KCLcurrentqNode_2)) ;
        }
    }

    return (OK) ;
}
