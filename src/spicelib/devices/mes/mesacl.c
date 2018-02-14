/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel*)inModel;
    MESinstance *here;
    double gdpr;
    double gspr;
    double gm;
    double gds;
    double ggs;
    double xgs;
    double ggd;
    double xgd;

    double m;

    for( ; model != NULL; model = MESnextModel(model)) {
        
        for( here = MESinstances(model); here != NULL; 
                here = MESnextInstance(here)) {

            m = here->MESm;

            gdpr=model->MESdrainConduct * here->MESarea;
            gspr=model->MESsourceConduct * here->MESarea;
            gm= *(ckt->CKTstate0 + here->MESgm) ;
            gds= *(ckt->CKTstate0 + here->MESgds) ;
            ggs= *(ckt->CKTstate0 + here->MESggs) ;
            xgs= *(ckt->CKTstate0 + here->MESqgs) * ckt->CKTomega ;
            ggd= *(ckt->CKTstate0 + here->MESggd) ;
            xgd= *(ckt->CKTstate0 + here->MESqgd) * ckt->CKTomega ;
            *(here->MESdrainDrainPtr )               += m * (gdpr);
            *(here->MESgateGatePtr )                 += m * (ggd+ggs);
            *(here->MESgateGatePtr +1)               += m * (xgd+xgs);
            *(here->MESsourceSourcePtr )             += m * (gspr);
            *(here->MESdrainPrimeDrainPrimePtr )     += m * (gdpr+gds+ggd);
            *(here->MESdrainPrimeDrainPrimePtr +1)   += m * (xgd);
            *(here->MESsourcePrimeSourcePrimePtr )   += m * (gspr+gds+gm+ggs);
            *(here->MESsourcePrimeSourcePrimePtr +1) += m * (xgs);
            *(here->MESdrainDrainPrimePtr )          -= m * (gdpr);
            *(here->MESgateDrainPrimePtr )           -= m * (ggd);
            *(here->MESgateDrainPrimePtr +1)         -= m * (xgd);
            *(here->MESgateSourcePrimePtr )          -= m * (ggs);
            *(here->MESgateSourcePrimePtr +1)        -= m * (xgs);
            *(here->MESsourceSourcePrimePtr )        -= m * (gspr);
            *(here->MESdrainPrimeDrainPtr )          -= m * (gdpr);
            *(here->MESdrainPrimeGatePtr )           += m * (-ggd+gm);
            *(here->MESdrainPrimeGatePtr +1)         -= m * (xgd);
            *(here->MESdrainPrimeSourcePrimePtr )    += m * (-gds-gm);
            *(here->MESsourcePrimeGatePtr )          += m * (-ggs-gm);
            *(here->MESsourcePrimeGatePtr +1)        -= m * (xgs);
            *(here->MESsourcePrimeSourcePtr )        -= m * (gspr);
            *(here->MESsourcePrimeDrainPrimePtr )    -= m * (gds);

        }
    }
    return(OK);
}
