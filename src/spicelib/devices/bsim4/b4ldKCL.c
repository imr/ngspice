/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

int
BSIM4loadKCL (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    double gdpr, gspr, gdtot, gstot, geltd, gcrg, m ;

    /* loop through all the BSIM4 models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            if (!model->BSIM4rdsMod)
            {
                gdpr = here->BSIM4drainConductance ;
                gspr = here->BSIM4sourceConductance ;
            } else {
                gdpr = gspr = 0.0 ;
            }

            if (model->BSIM4rdsMod == 1)
            {
                gdtot = here->BSIM4gdtot ;
                gstot = here->BSIM4gstot ;
            } else {
                gdtot = gstot = 0.0 ;
            }

            if (here->BSIM4rgateMod > 1)
            {
                gcrg = here->BSIM4gcrg ;
            } else {
                gcrg = 0.0 ;
            }

            geltd = here->BSIM4grgeltd ;
            m = here->BSIM4m ;

            /* KCL - Static Linear Part */
            *(ckt->CKTfvk+here->BSIM4dNodePrime) += m * (gdpr + gdtot) * (*(ckt->CKTrhsOld+here->BSIM4dNodePrime) - *(ckt->CKTrhsOld+here->BSIM4dNode)) ;
            *(here->KCLcurrentdNodePrime_1) = m * gdpr * (*(ckt->CKTrhsOld+here->BSIM4dNodePrime) - *(ckt->CKTrhsOld+here->BSIM4dNode)) ;
            *(here->KCLcurrentdNodePrime_2) = m * gdtot * (*(ckt->CKTrhsOld+here->BSIM4dNodePrime) - *(ckt->CKTrhsOld+here->BSIM4dNode)) ;

            *(ckt->CKTfvk+here->BSIM4dNode) -= m * (gdpr + gdtot) * (*(ckt->CKTrhsOld+here->BSIM4dNodePrime) - *(ckt->CKTrhsOld+here->BSIM4dNode)) ;
            *(here->KCLcurrentdNode_1) = -(m * gdpr * (*(ckt->CKTrhsOld+here->BSIM4dNodePrime) - *(ckt->CKTrhsOld+here->BSIM4dNode))) ;
            *(here->KCLcurrentdNode_2) = -(m * gdtot * (*(ckt->CKTrhsOld+here->BSIM4dNodePrime) - *(ckt->CKTrhsOld+here->BSIM4dNode))) ;

            *(ckt->CKTfvk+here->BSIM4sNodePrime) += m * (gspr + gstot) * (*(ckt->CKTrhsOld+here->BSIM4sNodePrime) - *(ckt->CKTrhsOld+here->BSIM4sNode)) ;
            *(here->KCLcurrentsNodePrime_1) = m * gspr * (*(ckt->CKTrhsOld+here->BSIM4sNodePrime) - *(ckt->CKTrhsOld+here->BSIM4sNode)) ;
            *(here->KCLcurrentsNodePrime_2) = m * gstot * (*(ckt->CKTrhsOld+here->BSIM4sNodePrime) - *(ckt->CKTrhsOld+here->BSIM4sNode)) ;

            *(ckt->CKTfvk+here->BSIM4sNode) -= m * (gspr + gstot) * (*(ckt->CKTrhsOld+here->BSIM4sNodePrime) - *(ckt->CKTrhsOld+here->BSIM4sNode)) ;
            *(here->KCLcurrentsNode_1) = -(m * gspr * (*(ckt->CKTrhsOld+here->BSIM4sNodePrime) - *(ckt->CKTrhsOld+here->BSIM4sNode))) ;
            *(here->KCLcurrentsNode_2) = -(m * gstot * (*(ckt->CKTrhsOld+here->BSIM4sNodePrime) - *(ckt->CKTrhsOld+here->BSIM4sNode))) ;

        if (here->BSIM4rgateMod == 1)
        {
            *(ckt->CKTfvk+here->BSIM4gNodeExt) += m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;
            *(here->KCLcurrentgNodeExt) = m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;

            *(ckt->CKTfvk+here->BSIM4gNodePrime) -= m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;
            *(here->KCLcurrentgNodePrime) = -(m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime))) ;
        } else if (here->BSIM4rgateMod == 2) {
            *(ckt->CKTfvk+here->BSIM4gNodeExt) += m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;
            *(here->KCLcurrentgNodeExt) = m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;

            *(ckt->CKTfvk+here->BSIM4gNodePrime) -= m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;
            *(here->KCLcurrentgNodePrime) = -(m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime))) ;
        } else if (here->BSIM4rgateMod == 3) {
            *(ckt->CKTfvk+here->BSIM4gNodeExt) += m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodeMid)) ;
            *(here->KCLcurrentgNodeExt) = m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodeMid)) ;

            *(ckt->CKTfvk+here->BSIM4gNodeMid) -= m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodeMid)) ;
            *(here->KCLcurrentgNodeMid_1) = -(m * geltd * (*(ckt->CKTrhsOld+here->BSIM4gNodeExt) - *(ckt->CKTrhsOld+here->BSIM4gNodeMid))) ;

            *(ckt->CKTfvk+here->BSIM4gNodeMid) += m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeMid) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;
            *(here->KCLcurrentgNodeMid_2) = m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeMid) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;

            *(ckt->CKTfvk+here->BSIM4gNodePrime) -= m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeMid) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime)) ;
            *(here->KCLcurrentgNodePrime) = -(m * gcrg * (*(ckt->CKTrhsOld+here->BSIM4gNodeMid) - *(ckt->CKTrhsOld+here->BSIM4gNodePrime))) ;
        }

        if (here->BSIM4rbodyMod)
        {
            *(ckt->CKTfvk+here->BSIM4dbNode) += m * here->BSIM4grbpd * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;
            *(here->KCLcurrentdbNode_1) = m * here->BSIM4grbpd * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;

            *(ckt->CKTfvk+here->BSIM4bNodePrime) -= m * here->BSIM4grbpd * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;
            *(here->KCLcurrentbNodePrime_1) = -(m * here->BSIM4grbpd * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime))) ;

            *(ckt->CKTfvk+here->BSIM4dbNode) += m * here->BSIM4grbdb * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode)) ;
            *(here->KCLcurrentdbNode_2) = m * here->BSIM4grbdb * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode)) ;

            *(ckt->CKTfvk+here->BSIM4bNode) -= m * here->BSIM4grbdb * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode)) ;
            *(here->KCLcurrentbNode_1) = -(m * here->BSIM4grbdb * (*(ckt->CKTrhsOld+here->BSIM4dbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode))) ;

            *(ckt->CKTfvk+here->BSIM4bNode) += m * here->BSIM4grbpb * (*(ckt->CKTrhsOld+here->BSIM4bNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;
            *(here->KCLcurrentbNode_2) = m * here->BSIM4grbpb * (*(ckt->CKTrhsOld+here->BSIM4bNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;

            *(ckt->CKTfvk+here->BSIM4bNodePrime) -= m * here->BSIM4grbpb * (*(ckt->CKTrhsOld+here->BSIM4bNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;
            *(here->KCLcurrentbNodePrime_2) = -(m * here->BSIM4grbpb * (*(ckt->CKTrhsOld+here->BSIM4bNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime))) ;

            *(ckt->CKTfvk+here->BSIM4sbNode) += m * here->BSIM4grbps * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;
            *(here->KCLcurrentsbNode_1) = m * here->BSIM4grbps * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;

            *(ckt->CKTfvk+here->BSIM4bNodePrime) -= m * here->BSIM4grbps * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime)) ;
            *(here->KCLcurrentbNodePrime_3) = -(m * here->BSIM4grbps * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNodePrime))) ;

            *(ckt->CKTfvk+here->BSIM4sbNode) += m * here->BSIM4grbsb * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode)) ;
            *(here->KCLcurrentsbNode_2) = m * here->BSIM4grbsb * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode)) ;

            *(ckt->CKTfvk+here->BSIM4bNode) -= m * here->BSIM4grbsb * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode)) ;
            *(here->KCLcurrentbNode_3) = -(m * here->BSIM4grbsb * (*(ckt->CKTrhsOld+here->BSIM4sbNode) - *(ckt->CKTrhsOld+here->BSIM4bNode))) ;
            }
        }
    }

    return (OK) ;
}
