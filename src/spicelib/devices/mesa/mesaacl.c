/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "mesadefs.h"
#include "sperror.h"
#include "suffix.h"


int
MESAacLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
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

    for( ; model != NULL; model = model->MESAnextModel ) {
        for( here = model->MESAinstances; here != NULL; 
             here = here->MESAnextInstance) {
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
            
            *(here->MESAdrainDrainPtr) += here->MESAdrainConduct;
            *(here->MESAsourceSourcePtr) += here->MESAsourceConduct;
            *(here->MESAgateGatePtr) += here->MESAgateConduct;
            *(here->MESAsourcePrmPrmSourcePrmPrmPtr) += (here->MESAtGi+ggspp);
            *(here->MESAdrainPrmPrmDrainPrmPrmPtr) += (here->MESAtGf+ggdpp);
            *(here->MESAdrainDrainPrimePtr) -= here->MESAdrainConduct;
            *(here->MESAdrainPrimeDrainPtr) -= here->MESAdrainConduct;
            *(here->MESAsourceSourcePrimePtr) -= here->MESAsourceConduct;
            *(here->MESAsourcePrimeSourcePtr) -= here->MESAsourceConduct;
            *(here->MESAgateGatePrimePtr) -= here->MESAgateConduct;
            *(here->MESAgatePrimeGatePtr) -= here->MESAgateConduct;
            *(here->MESAgatePrimeDrainPrimePtr) += (-ggd);
            *(here->MESAgatePrimeSourcePrimePtr) += (-ggs);
            *(here->MESAdrainPrimeGatePrimePtr) += (gm-ggd);
            *(here->MESAdrainPrimeSourcePrimePtr) += (-gds-gm);
            *(here->MESAsourcePrimeGatePrimePtr) += (-ggs-gm);
            *(here->MESAsourcePrimeDrainPrimePtr) += (-gds);
            *(here->MESAgatePrimeGatePrimePtr) += (ggd+ggs+here->MESAgateConduct+ggspp+ggdpp);
            *(here->MESAdrainPrimeDrainPrimePtr) += (gds+ggd+here->MESAdrainConduct+here->MESAtGf);
            *(here->MESAsourcePrimeSourcePrimePtr) += (gds+gm+ggs+here->MESAsourceConduct+here->MESAtGi);
            *(here->MESAsourcePrimeSourcePrmPrmPtr) -= here->MESAtGi;
            *(here->MESAsourcePrmPrmSourcePrimePtr) -= here->MESAtGi;
            *(here->MESAgatePrimeSourcePrmPrmPtr) -= ggspp;
            *(here->MESAsourcePrmPrmGatePrimePtr) -= ggspp;
            *(here->MESAdrainPrimeDrainPrmPrmPtr) -= here->MESAtGf;
            *(here->MESAdrainPrmPrmDrainPrimePtr) -= here->MESAtGf;
            *(here->MESAgatePrimeDrainPrmPrmPtr) -= ggdpp;
            *(here->MESAdrainPrmPrmGatePrimePtr) -= ggdpp;
            *(here->MESAsourcePrmPrmSourcePrmPrmPtr+1) += xgs;
            *(here->MESAdrainPrmPrmDrainPrmPrmPtr+1) += xgd;
            *(here->MESAgatePrimeGatePrimePtr+1) += xgd+xgs;
            *(here->MESAgatePrimeDrainPrmPrmPtr+1) -= xgd;
            *(here->MESAdrainPrmPrmGatePrimePtr+1) -= xgd;
            *(here->MESAgatePrimeSourcePrmPrmPtr+1) -= xgs;
            *(here->MESAsourcePrmPrmGatePrimePtr+1) -= xgs;
        }
    }
    return(OK);
}
