/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "jfetdefs.h"
#include "suffix.h"


int
JFETpzLoad(inModel,ckt,s)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    register SPcomplex *s;
{
    register JFETmodel *model = (JFETmodel*)inModel;
    register JFETinstance *here;
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

            gdpr=model->JFETdrainResist * here->JFETarea;
            gspr=model->JFETsourceResist * here->JFETarea;
            gm= *(ckt->CKTstate0 + here->JFETgm) ;
            gds= *(ckt->CKTstate0 + here->JFETgds) ;
            ggs= *(ckt->CKTstate0 + here->JFETggs) ;
            xgs= *(ckt->CKTstate0 + here->JFETqgs) ;
            ggd= *(ckt->CKTstate0 + here->JFETggd) ;
            xgd= *(ckt->CKTstate0 + here->JFETqgd) ;
            *(here->JFETdrainDrainPtr ) += gdpr;
            *(here->JFETgateGatePtr ) += ggd+ggs;
            *(here->JFETgateGatePtr   ) += (xgd+xgs) * s->real;
            *(here->JFETgateGatePtr +1) += (xgd+xgs) * s->imag;
            *(here->JFETsourceSourcePtr ) += gspr;
            *(here->JFETdrainPrimeDrainPrimePtr ) += gdpr+gds+ggd;
            *(here->JFETdrainPrimeDrainPrimePtr   ) += xgd * s->real;
            *(here->JFETdrainPrimeDrainPrimePtr +1) += xgd * s->imag;
            *(here->JFETsourcePrimeSourcePrimePtr ) += gspr+gds+gm+ggs;
            *(here->JFETsourcePrimeSourcePrimePtr   ) += xgs * s->real;
            *(here->JFETsourcePrimeSourcePrimePtr +1) += xgs * s->imag;
            *(here->JFETdrainDrainPrimePtr ) -= gdpr;
            *(here->JFETgateDrainPrimePtr ) -= ggd;
            *(here->JFETgateDrainPrimePtr   ) -= xgd * s->real;
            *(here->JFETgateDrainPrimePtr +1) -= xgd * s->imag;
            *(here->JFETgateSourcePrimePtr ) -= ggs;
            *(here->JFETgateSourcePrimePtr   ) -= xgs * s->real;
            *(here->JFETgateSourcePrimePtr +1) -= xgs * s->imag;
            *(here->JFETsourceSourcePrimePtr ) -= gspr;
            *(here->JFETdrainPrimeDrainPtr ) -= gdpr;
            *(here->JFETdrainPrimeGatePtr ) += (-ggd+gm);
            *(here->JFETdrainPrimeGatePtr   ) -= xgd * s->real;
            *(here->JFETdrainPrimeGatePtr +1) -= xgd * s->imag;
            *(here->JFETdrainPrimeSourcePrimePtr ) += (-gds-gm);
            *(here->JFETsourcePrimeGatePtr ) += (-ggs-gm);
            *(here->JFETsourcePrimeGatePtr   ) -= xgs * s->real;
            *(here->JFETsourcePrimeGatePtr +1) -= xgs * s->imag;
            *(here->JFETsourcePrimeSourcePtr ) -= gspr;
            *(here->JFETsourcePrimeDrainPrimePtr ) -= gds;

        }
    }
    return(OK);
}
