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
#include "complex.h"
#include "suffix.h"


int
MESpzLoad(inModel,ckt,s)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    SPcomplex *s;
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
            xgs= *(ckt->CKTstate0 + here->MESqgs) ;
            ggd= *(ckt->CKTstate0 + here->MESggd) ;
            xgd= *(ckt->CKTstate0 + here->MESqgd) ;
            *(here->MESdrainDrainPtr ) += gdpr;
            *(here->MESgateGatePtr ) += ggd+ggs;
            *(here->MESgateGatePtr   ) += (xgd+xgs)*s->real;
            *(here->MESgateGatePtr +1) += (xgd+xgs)*s->imag;
            *(here->MESsourceSourcePtr ) += gspr;
            *(here->MESdrainPrimeDrainPrimePtr ) += gdpr+gds+ggd;
            *(here->MESdrainPrimeDrainPrimePtr   ) += xgd*s->real;
            *(here->MESdrainPrimeDrainPrimePtr +1) += xgd*s->imag;
            *(here->MESsourcePrimeSourcePrimePtr ) += gspr+gds+gm+ggs;
            *(here->MESsourcePrimeSourcePrimePtr   ) += xgs*s->real;
            *(here->MESsourcePrimeSourcePrimePtr +1) += xgs*s->imag;
            *(here->MESdrainDrainPrimePtr ) -= gdpr;
            *(here->MESgateDrainPrimePtr ) -= ggd;
            *(here->MESgateDrainPrimePtr   ) -= xgd*s->real;
            *(here->MESgateDrainPrimePtr +1) -= xgd*s->imag;
            *(here->MESgateSourcePrimePtr ) -= ggs;
            *(here->MESgateSourcePrimePtr   ) -= xgs*s->real;
            *(here->MESgateSourcePrimePtr +1) -= xgs*s->imag;
            *(here->MESsourceSourcePrimePtr ) -= gspr;
            *(here->MESdrainPrimeDrainPtr ) -= gdpr;
            *(here->MESdrainPrimeGatePtr ) += (-ggd+gm);
            *(here->MESdrainPrimeGatePtr   ) -= xgd*s->real;
            *(here->MESdrainPrimeGatePtr +1) -= xgd*s->imag;
            *(here->MESdrainPrimeSourcePrimePtr ) += (-gds-gm);
            *(here->MESsourcePrimeGatePtr ) += (-ggs-gm);
            *(here->MESsourcePrimeGatePtr   ) -= xgs*s->real;
            *(here->MESsourcePrimeGatePtr +1) -= xgs*s->imag;
            *(here->MESsourcePrimeSourcePtr ) -= gspr;
            *(here->MESsourcePrimeDrainPrimePtr ) -= gds;

        }
    }
    return(OK);
}
