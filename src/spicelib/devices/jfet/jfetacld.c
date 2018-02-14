/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
JFETacLoad(GENmodel *inModel, CKTcircuit *ckt)
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

    double m;

    for( ; model != NULL; model = JFETnextModel(model)) {
        
        for( here = JFETinstances(model); here != NULL; 
                here = JFETnextInstance(here)) {

            gdpr=model->JFETdrainConduct * here->JFETarea;
            gspr=model->JFETsourceConduct * here->JFETarea;
            gm= *(ckt->CKTstate0 + here->JFETgm) ;
            gds= *(ckt->CKTstate0 + here->JFETgds) ;
            ggs= *(ckt->CKTstate0 + here->JFETggs) ;
            xgs= *(ckt->CKTstate0 + here->JFETqgs) * ckt->CKTomega ;
            ggd= *(ckt->CKTstate0 + here->JFETggd) ;
            xgd= *(ckt->CKTstate0 + here->JFETqgd) * ckt->CKTomega ;
            
            m = here->JFETm;

            *(here->JFETdrainDrainPtr )               += m * (gdpr);
            *(here->JFETgateGatePtr )                 += m * (ggd+ggs);
            *(here->JFETgateGatePtr +1)               += m * (xgd+xgs);
            *(here->JFETsourceSourcePtr )             += m * (gspr);
            *(here->JFETdrainPrimeDrainPrimePtr )     += m * (gdpr+gds+ggd);
            *(here->JFETdrainPrimeDrainPrimePtr +1)   += m * (xgd);
            *(here->JFETsourcePrimeSourcePrimePtr )   += m * (gspr+gds+gm+ggs);
            *(here->JFETsourcePrimeSourcePrimePtr +1) += m * (xgs);
            *(here->JFETdrainDrainPrimePtr )          -= m * (gdpr);
            *(here->JFETgateDrainPrimePtr )           -= m * (ggd);
            *(here->JFETgateDrainPrimePtr +1)         -= m * (xgd);
            *(here->JFETgateSourcePrimePtr )          -= m * (ggs);
            *(here->JFETgateSourcePrimePtr +1)        -= m * (xgs);
            *(here->JFETsourceSourcePrimePtr )        -= m * (gspr);
            *(here->JFETdrainPrimeDrainPtr )          -= m * (gdpr);
            *(here->JFETdrainPrimeGatePtr )           += m * (-ggd+gm);
            *(here->JFETdrainPrimeGatePtr +1)         -= m * (xgd);
            *(here->JFETdrainPrimeSourcePrimePtr )    += m * (-gds-gm);
            *(here->JFETsourcePrimeGatePtr )          += m * (-ggs-gm);
            *(here->JFETsourcePrimeGatePtr +1)        -= m * (xgs);
            *(here->JFETsourcePrimeSourcePtr )        -= m * (gspr);
            *(here->JFETsourcePrimeDrainPrimePtr )    -= m * (gds);

        }
    }
    return(OK);
}
