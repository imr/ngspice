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


            /* KCL - Non-Linear and Dynamic Linear Parts */
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrimeRHS_1)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrimeRHS_2)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrimeRHS_3)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrimeRHS_4)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrimeRHS_5)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4gNodePrime, &(here->KCLcurrentgNodePrimeRHS_1)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4gNodePrime, &(here->KCLcurrentgNodePrimeRHS_2)) ;

            if (here->BSIM4rgateMod == 3)
                error = CKTmkCurKCL (ckt, here->BSIM4gNodeMid, &(here->KCLcurrentgNodeMidRHS)) ;

            if (!here->BSIM4rbodyMod)
            {
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_5)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_6)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_5)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_6)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_7)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_8)) ;
            } else {
                error = CKTmkCurKCL (ckt, here->BSIM4dbNode, &(here->KCLcurrentdbNodeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4dbNode, &(here->KCLcurrentdbNodeRHS_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4bNodePrime, &(here->KCLcurrentbNodePrimeRHS_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sbNode, &(here->KCLcurrentsbNodeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sbNode, &(here->KCLcurrentsbNodeRHS_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_2)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_3)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_4)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_5)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_6)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_7)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_8)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_9)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrimeRHS_10)) ;
            }

            if (here->BSIM4trnqsMod)
                error = CKTmkCurKCL (ckt, here->BSIM4qNode, &(here->KCLcurrentqNodeRHS_1)) ;
                error = CKTmkCurKCL (ckt, here->BSIM4qNode, &(here->KCLcurrentqNodeRHS_2)) ;


            /* KCL - Static Linear Part */
            error = CKTmkCurKCL (ckt, here->BSIM4dNodePrime, &(here->KCLcurrentdNodePrime)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4dNode, &(here->KCLcurrentdNode)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4sNodePrime, &(here->KCLcurrentsNodePrime)) ;
            error = CKTmkCurKCL (ckt, here->BSIM4sNode, &(here->KCLcurrentsNode)) ;
        }
    }

    return (OK) ;
}
