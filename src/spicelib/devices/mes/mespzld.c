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
#include "ngspice/complex.h"
#include "ngspice/suffix.h"


int
MESpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
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
            xgs= *(ckt->CKTstate0 + here->MESqgs) ;
            ggd= *(ckt->CKTstate0 + here->MESggd) ;
            xgd= *(ckt->CKTstate0 + here->MESqgd) ;

            *(here->MESdrainDrainPtr )               += m * gdpr;
            *(here->MESgateGatePtr )                 += m * ggd+ggs;
            *(here->MESgateGatePtr   )               += m * (xgd+xgs)*s->real;
            *(here->MESgateGatePtr +1)               += m * (xgd+xgs)*s->imag;
            *(here->MESsourceSourcePtr )             += m * gspr;
            *(here->MESdrainPrimeDrainPrimePtr )     += m * gdpr+gds+ggd;
            *(here->MESdrainPrimeDrainPrimePtr   )   += m * xgd*s->real;
            *(here->MESdrainPrimeDrainPrimePtr +1)   += m * xgd*s->imag;
            *(here->MESsourcePrimeSourcePrimePtr )   += m * gspr+gds+gm+ggs;
            *(here->MESsourcePrimeSourcePrimePtr   ) += m * xgs*s->real;
            *(here->MESsourcePrimeSourcePrimePtr +1) += m * xgs*s->imag;
            *(here->MESdrainDrainPrimePtr )          -= m * gdpr;
            *(here->MESgateDrainPrimePtr )           -= m * ggd;
            *(here->MESgateDrainPrimePtr   )         -= m * xgd*s->real;
            *(here->MESgateDrainPrimePtr +1)         -= m * xgd*s->imag;
            *(here->MESgateSourcePrimePtr )          -= m * ggs;
            *(here->MESgateSourcePrimePtr   )        -= m * xgs*s->real;
            *(here->MESgateSourcePrimePtr +1)        -= m * xgs*s->imag;
            *(here->MESsourceSourcePrimePtr )        -= m * gspr;
            *(here->MESdrainPrimeDrainPtr )          -= m * gdpr;
            *(here->MESdrainPrimeGatePtr )           += m * (-ggd+gm);
            *(here->MESdrainPrimeGatePtr   )         -= m * xgd*s->real;
            *(here->MESdrainPrimeGatePtr +1)         -= m * xgd*s->imag;
            *(here->MESdrainPrimeSourcePrimePtr )    += m * (-gds-gm);
            *(here->MESsourcePrimeGatePtr )          += m * (-ggs-gm);
            *(here->MESsourcePrimeGatePtr   )        -= m * xgs*s->real;
            *(here->MESsourcePrimeGatePtr +1)        -= m * xgs*s->imag;
            *(here->MESsourcePrimeSourcePtr )        -= m * gspr;
            *(here->MESsourcePrimeDrainPrimePtr )    -= m * gds;

        }
    }
    return(OK);
}
