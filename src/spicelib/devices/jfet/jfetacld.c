/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "jfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
JFETacLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
    JFETmodel *model = (JFETmodel*)inModel;
    JFETinstance *here;
    double gdpr;
    double gspr;
    double gm;
    double gds;
    double ggs;
    double xgs;
    double ggd;
    double xgd;

    for( ; model != NULL; model = model->JFETnextModel ) {
        
        for( here = model->JFETinstances; here != NULL; 
                here = here->JFETnextInstance) {
	    if (here->JFETowner != ARCHme) continue;

            gdpr=model->JFETdrainConduct * here->JFETarea;
            gspr=model->JFETsourceConduct * here->JFETarea;
            gm= *(ckt->CKTstate0 + here->JFETgm) ;
            gds= *(ckt->CKTstate0 + here->JFETgds) ;
            ggs= *(ckt->CKTstate0 + here->JFETggs) ;
            xgs= *(ckt->CKTstate0 + here->JFETqgs) * ckt->CKTomega ;
            ggd= *(ckt->CKTstate0 + here->JFETggd) ;
            xgd= *(ckt->CKTstate0 + here->JFETqgd) * ckt->CKTomega ;
            *(here->JFETdrainDrainPtr ) += gdpr;
            *(here->JFETgateGatePtr ) += ggd+ggs;
            *(here->JFETgateGatePtr +1) += xgd+xgs;
            *(here->JFETsourceSourcePtr ) += gspr;
            *(here->JFETdrainPrimeDrainPrimePtr ) += gdpr+gds+ggd;
            *(here->JFETdrainPrimeDrainPrimePtr +1) += xgd;
            *(here->JFETsourcePrimeSourcePrimePtr ) += gspr+gds+gm+ggs;
            *(here->JFETsourcePrimeSourcePrimePtr +1) += xgs;
            *(here->JFETdrainDrainPrimePtr ) -= gdpr;
            *(here->JFETgateDrainPrimePtr ) -= ggd;
            *(here->JFETgateDrainPrimePtr +1) -= xgd;
            *(here->JFETgateSourcePrimePtr ) -= ggs;
            *(here->JFETgateSourcePrimePtr +1) -= xgs;
            *(here->JFETsourceSourcePrimePtr ) -= gspr;
            *(here->JFETdrainPrimeDrainPtr ) -= gdpr;
            *(here->JFETdrainPrimeGatePtr ) += (-ggd+gm);
            *(here->JFETdrainPrimeGatePtr +1) -= xgd;
            *(here->JFETdrainPrimeSourcePrimePtr ) += (-gds-gm);
            *(here->JFETsourcePrimeGatePtr ) += (-ggs-gm);
            *(here->JFETsourcePrimeGatePtr +1) -= xgs;
            *(here->JFETsourcePrimeSourcePtr ) -= gspr;
            *(here->JFETsourcePrimeDrainPrimePtr ) -= gds;

        }
    }
    return(OK);
}
