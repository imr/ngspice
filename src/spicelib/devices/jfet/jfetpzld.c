/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "jfetdefs.h"
#include "ngspice/suffix.h"


int
JFETpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
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

            m = here->JFETm;

            gdpr=model->JFETdrainConduct * here->JFETarea;
            gspr=model->JFETsourceConduct * here->JFETarea;
            gm= *(ckt->CKTstate0 + here->JFETgm) ;
            gds= *(ckt->CKTstate0 + here->JFETgds) ;
            ggs= *(ckt->CKTstate0 + here->JFETggs) ;
            xgs= *(ckt->CKTstate0 + here->JFETqgs) ;
            ggd= *(ckt->CKTstate0 + here->JFETggd) ;
            xgd= *(ckt->CKTstate0 + here->JFETqgd) ;

            *(here->JFETdrainDrainPtr )               += m * gdpr;
            *(here->JFETgateGatePtr )                 += m * (ggd+ggs);
            *(here->JFETgateGatePtr   )               += m * ((xgd+xgs) * s->real);
            *(here->JFETgateGatePtr +1)               += m * ((xgd+xgs) * s->imag);
            *(here->JFETsourceSourcePtr )             += m * (gspr);
            *(here->JFETdrainPrimeDrainPrimePtr )     += m * (gdpr+gds+ggd);
            *(here->JFETdrainPrimeDrainPrimePtr   )   += m * (xgd * s->real);
            *(here->JFETdrainPrimeDrainPrimePtr +1)   += m * (xgd * s->imag);
            *(here->JFETsourcePrimeSourcePrimePtr )   += m * (gspr+gds+gm+ggs);
            *(here->JFETsourcePrimeSourcePrimePtr   ) += m * (xgs * s->real);
            *(here->JFETsourcePrimeSourcePrimePtr +1) += m * (xgs * s->imag);
            *(here->JFETdrainDrainPrimePtr )          -= m * (gdpr);
            *(here->JFETgateDrainPrimePtr )           -= m * (ggd);
            *(here->JFETgateDrainPrimePtr   )         -= m * (xgd * s->real);
            *(here->JFETgateDrainPrimePtr +1)         -= m * (xgd * s->imag);
            *(here->JFETgateSourcePrimePtr )          -= m * (ggs);
            *(here->JFETgateSourcePrimePtr   )        -= m * (xgs * s->real);
            *(here->JFETgateSourcePrimePtr +1)        -= m * (xgs * s->imag);
            *(here->JFETsourceSourcePrimePtr )        -= m * (gspr);
            *(here->JFETdrainPrimeDrainPtr )          -= m * (gdpr);
            *(here->JFETdrainPrimeGatePtr )           += m * (-ggd+gm);
            *(here->JFETdrainPrimeGatePtr   )         -= m * (xgd * s->real);
            *(here->JFETdrainPrimeGatePtr +1)         -= m * (xgd * s->imag);
            *(here->JFETdrainPrimeSourcePrimePtr )    += m * (-gds-gm);
            *(here->JFETsourcePrimeGatePtr )          += m * (-ggs-gm);
            *(here->JFETsourcePrimeGatePtr   )        -= m * (xgs * s->real);
            *(here->JFETsourcePrimeGatePtr +1)        -= m * (xgs * s->imag);
            *(here->JFETsourcePrimeSourcePtr )        -= m * (gspr);
            *(here->JFETsourcePrimeDrainPrimePtr )    -= m * (gds);

        }
    }
    return(OK);
}
