/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "mesdefs.h"
#include "sperror.h"
#include "suffix.h"


int
MESacLoad(inModel,ckt)
    GENmodel *inModel;
    register CKTcircuit *ckt;
{
    register MESmodel *model = (MESmodel*)inModel;
    register MESinstance *here;
    double gdpr;
    double gspr;
    double gm;
    double gds;
    double ggs;
    double xgs;
    double ggd;
    double xgd;

    for( ; model != NULL; model = model->MESnextModel ) {
        
        for( here = model->MESinstances; here != NULL; 
                here = here->MESnextInstance) {
	    if (here->MESowner != ARCHme) continue;

            gdpr=model->MESdrainConduct * here->MESarea;
            gspr=model->MESsourceConduct * here->MESarea;
            gm= *(ckt->CKTstate0 + here->MESgm) ;
            gds= *(ckt->CKTstate0 + here->MESgds) ;
            ggs= *(ckt->CKTstate0 + here->MESggs) ;
            xgs= *(ckt->CKTstate0 + here->MESqgs) * ckt->CKTomega ;
            ggd= *(ckt->CKTstate0 + here->MESggd) ;
            xgd= *(ckt->CKTstate0 + here->MESqgd) * ckt->CKTomega ;
            *(here->MESdrainDrainPtr ) += gdpr;
            *(here->MESgateGatePtr ) += ggd+ggs;
            *(here->MESgateGatePtr +1) += xgd+xgs;
            *(here->MESsourceSourcePtr ) += gspr;
            *(here->MESdrainPrimeDrainPrimePtr ) += gdpr+gds+ggd;
            *(here->MESdrainPrimeDrainPrimePtr +1) += xgd;
            *(here->MESsourcePrimeSourcePrimePtr ) += gspr+gds+gm+ggs;
            *(here->MESsourcePrimeSourcePrimePtr +1) += xgs;
            *(here->MESdrainDrainPrimePtr ) -= gdpr;
            *(here->MESgateDrainPrimePtr ) -= ggd;
            *(here->MESgateDrainPrimePtr +1) -= xgd;
            *(here->MESgateSourcePrimePtr ) -= ggs;
            *(here->MESgateSourcePrimePtr +1) -= xgs;
            *(here->MESsourceSourcePrimePtr ) -= gspr;
            *(here->MESdrainPrimeDrainPtr ) -= gdpr;
            *(here->MESdrainPrimeGatePtr ) += (-ggd+gm);
            *(here->MESdrainPrimeGatePtr +1) -= xgd;
            *(here->MESdrainPrimeSourcePrimePtr ) += (-gds-gm);
            *(here->MESsourcePrimeGatePtr ) += (-ggs-gm);
            *(here->MESsourcePrimeGatePtr +1) -= xgs;
            *(here->MESsourcePrimeSourcePtr ) -= gspr;
            *(here->MESsourcePrimeDrainPrimePtr ) -= gds;

        }
    }
    return(OK);
}
