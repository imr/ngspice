/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESAacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel*)inModel;
    MESAinstance *here;
    double gm;
    double gds;
    double ggspp;
    double ggdpp;
    double ggs;
    double xgs;
    double ggd;
    double xgd;
    double f;
    double lambda;
    double vds;
    double delidgch;
    double delidvds;

    double m;

    for( ; model != NULL; model = MESAnextModel(model)) {
        for( here = MESAinstances(model); here != NULL; 
             here = MESAnextInstance(here)) {

            f      = ckt->CKTomega/2/M_PI;
            if(here->MESAdelf == 0)
              lambda = here->MESAtLambda;
            else
              lambda = here->MESAtLambda+0.5*(here->MESAtLambdahf-here->MESAtLambda)*
                       (1+tanh((f-here->MESAfl)/here->MESAdelf));
            vds= *(ckt->CKTstate0 + here->MESAvgs) -
                 *(ckt->CKTstate0 + here->MESAvgd);
            delidgch = here->MESAdelidgch0*(1+lambda*vds);
            delidvds = here->MESAdelidvds0*(1+2*lambda*vds) -
                       here->MESAdelidvds1;
            gm       = (delidgch*here->MESAgm0+here->MESAgm1)*here->MESAgm2;
            gds      = delidvds+here->MESAgds0;

            ggspp=*(ckt->CKTstate0 + here->MESAggspp);
            ggdpp=*(ckt->CKTstate0 + here->MESAggdpp);
            ggs= *(ckt->CKTstate0 + here->MESAggs) ;
            xgs= *(ckt->CKTstate0 + here->MESAqgs) * ckt->CKTomega ;
            ggd= *(ckt->CKTstate0 + here->MESAggd) ;
            xgd= *(ckt->CKTstate0 + here->MESAqgd) * ckt->CKTomega ;
            
            m = here->MESAm;

            *(here->MESAdrainDrainPtr)                 += m * (here->MESAdrainConduct);
            *(here->MESAsourceSourcePtr)               += m * (here->MESAsourceConduct);
            *(here->MESAgateGatePtr)                   += m * (here->MESAgateConduct);
            *(here->MESAsourcePrmPrmSourcePrmPrmPtr)   += m * (here->MESAtGi+ggspp);
            *(here->MESAdrainPrmPrmDrainPrmPrmPtr)     += m * (here->MESAtGf+ggdpp);
            *(here->MESAdrainDrainPrimePtr)            -= m * (here->MESAdrainConduct);
            *(here->MESAdrainPrimeDrainPtr)            -= m * (here->MESAdrainConduct);
            *(here->MESAsourceSourcePrimePtr)          -= m * (here->MESAsourceConduct);
            *(here->MESAsourcePrimeSourcePtr)          -= m * (here->MESAsourceConduct);
            *(here->MESAgateGatePrimePtr)              -= m * (here->MESAgateConduct);
            *(here->MESAgatePrimeGatePtr)              -= m * (here->MESAgateConduct);
            *(here->MESAgatePrimeDrainPrimePtr)        += m * (-ggd);
            *(here->MESAgatePrimeSourcePrimePtr)       += m * (-ggs);
            *(here->MESAdrainPrimeGatePrimePtr)        += m * (gm-ggd);
            *(here->MESAdrainPrimeSourcePrimePtr)      += m * (-gds-gm);
            *(here->MESAsourcePrimeGatePrimePtr)       += m * (-ggs-gm);
            *(here->MESAsourcePrimeDrainPrimePtr)      += m * (-gds);
            *(here->MESAgatePrimeGatePrimePtr)         += m * (ggd+ggs+here->MESAgateConduct+ggspp+ggdpp);
            *(here->MESAdrainPrimeDrainPrimePtr)       += m * (gds+ggd+here->MESAdrainConduct+here->MESAtGf);
            *(here->MESAsourcePrimeSourcePrimePtr)     += m * (gds+gm+ggs+here->MESAsourceConduct+here->MESAtGi);
            *(here->MESAsourcePrimeSourcePrmPrmPtr)    -= m * (here->MESAtGi);
            *(here->MESAsourcePrmPrmSourcePrimePtr)    -= m * (here->MESAtGi);
            *(here->MESAgatePrimeSourcePrmPrmPtr)      -= m * (ggspp);
            *(here->MESAsourcePrmPrmGatePrimePtr)      -= m * (ggspp);
            *(here->MESAdrainPrimeDrainPrmPrmPtr)      -= m * (here->MESAtGf);
            *(here->MESAdrainPrmPrmDrainPrimePtr)      -= m * (here->MESAtGf);
            *(here->MESAgatePrimeDrainPrmPrmPtr)       -= m * (ggdpp);
            *(here->MESAdrainPrmPrmGatePrimePtr)       -= m * (ggdpp);
            *(here->MESAsourcePrmPrmSourcePrmPrmPtr+1) += m * (xgs);
            *(here->MESAdrainPrmPrmDrainPrmPrmPtr+1)   += m * (xgd);
            *(here->MESAgatePrimeGatePrimePtr+1)       += m * (xgd+xgs);
            *(here->MESAgatePrimeDrainPrmPrmPtr+1)     -= m * (xgd);
            *(here->MESAdrainPrmPrmGatePrimePtr+1)     -= m * (xgd);
            *(here->MESAgatePrimeSourcePrmPrmPtr+1)    -= m * (xgs);
            *(here->MESAsourcePrmPrmGatePrimePtr+1)    -= m * (xgs);
        }
    }
    return(OK);
}
