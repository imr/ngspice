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
        }
    }

    return (OK) ;
}
