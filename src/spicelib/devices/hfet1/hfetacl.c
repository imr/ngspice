
#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
HFETAacLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;
    double gm;
    double gds;
    double xds;
    double ggs;
    double xgs;
    double ggd;
    double xgd;
    double ggspp;
    double ggdpp;
    double f;

    for( ; model != NULL; model = model->HFETAnextModel ) 
    {
        for( here = model->HFETAinstances; here != NULL; 
             here = here->HFETAnextInstance) 
        {
            gm  = *(ckt->CKTstate0 + here->HFETAgm);
            gds = *(ckt->CKTstate0 + here->HFETAgds);
            xds = CDS*ckt->CKTomega;
            ggs = *(ckt->CKTstate0 + here->HFETAggs);
            xgs = *(ckt->CKTstate0 + here->HFETAqgs) * ckt->CKTomega;
            ggd = *(ckt->CKTstate0 + here->HFETAggd);
            xgd = *(ckt->CKTstate0 + here->HFETAqgd) * ckt->CKTomega;
            ggspp = *(ckt->CKTstate0 + here->HFETAggspp);
            ggdpp = *(ckt->CKTstate0 + here->HFETAggdpp);
            if(model->HFETAkappaGiven && here->HFETAdelf != 0.0) {
              f     = ckt->CKTomega/2/M_PI;
              gds   = gds*(1+0.5*model->HFETAkappa*(1+tanh((f-here->HFETAfgds)/here->HFETAdelf)));
            }  
            *(here->HFETAdrainDrainPtr) += model->HFETAdrainConduct;
            *(here->HFETAsourceSourcePtr) += model->HFETAsourceConduct;
            *(here->HFETAgatePrimeGatePrimePtr) += (ggd+ggs+ggspp+ggdpp+model->HFETAgateConduct);
            *(here->HFETAdrainPrimeDrainPrimePtr) += (gds+ggd+model->HFETAdrainConduct+model->HFETAgf);
            *(here->HFETAsourcePrimeSourcePrimePtr) += (gds+gm+ggs+model->HFETAsourceConduct+model->HFETAgi);
            *(here->HFETAsourcePrmPrmSourcePrmPrmPtr) += (model->HFETAgi+ggspp);
            *(here->HFETAdrainPrmPrmDrainPrmPrmPtr) += (model->HFETAgf+ggdpp);
            *(here->HFETAdrainDrainPrimePtr) -= model->HFETAdrainConduct;
            *(here->HFETAdrainPrimeDrainPtr) -= model->HFETAdrainConduct;
            *(here->HFETAsourceSourcePrimePtr) -= model->HFETAsourceConduct;
            *(here->HFETAsourcePrimeSourcePtr) -= model->HFETAsourceConduct;
            *(here->HFETAgatePrimeDrainPrimePtr) -= ggd;
            *(here->HFETAdrainPrimeGatePrimePtr) += (gm-ggd);
            *(here->HFETAgatePrimeSourcePrimePtr) -= ggs;
            *(here->HFETAsourcePrimeGatePrimePtr) += (-ggs-gm);
            *(here->HFETAdrainPrimeSourcePrimePtr) += (-gds-gm);
            *(here->HFETAsourcePrimeDrainPrimePtr) -= gds;
            *(here->HFETAsourcePrimeSourcePrmPrmPtr) -= model->HFETAgi;            
            *(here->HFETAsourcePrmPrmSourcePrimePtr) -= model->HFETAgi;
            *(here->HFETAgatePrimeSourcePrmPrmPtr) -= ggspp;
            *(here->HFETAsourcePrmPrmGatePrimePtr) -= ggspp;
            *(here->HFETAdrainPrimeDrainPrmPrmPtr) -= model->HFETAgf;
            *(here->HFETAdrainPrmPrmDrainPrimePtr) -= model->HFETAgf;
            *(here->HFETAgatePrimeDrainPrmPrmPtr) -= ggdpp;
            *(here->HFETAdrainPrmPrmGatePrimePtr) -= ggdpp;
            *(here->HFETAgateGatePtr) += model->HFETAgateConduct;
            *(here->HFETAgateGatePrimePtr) -= model->HFETAgateConduct;
            *(here->HFETAgatePrimeGatePtr) -= model->HFETAgateConduct;
            *(here->HFETAgatePrimeGatePrimePtr+1) += xgd+xgs;                                       
            *(here->HFETAdrainPrmPrmDrainPrmPrmPtr+1) += xgd;
            *(here->HFETAsourcePrmPrmSourcePrmPrmPtr+1) += xgs;
            *(here->HFETAgatePrimeDrainPrmPrmPtr+1) -= xgd;
            *(here->HFETAgatePrimeSourcePrmPrmPtr+1) -= xgs;
            *(here->HFETAdrainPrmPrmGatePrimePtr+1) -= xgd;
            *(here->HFETAsourcePrmPrmGatePrimePtr+1) -= xgs;
            *(here->HFETAdrainPrimeDrainPrimePtr+1) += xds;
            *(here->HFETAsourcePrimeSourcePrimePtr+1) += xds;
            *(here->HFETAdrainPrimeSourcePrimePtr+1) -= xds;
            *(here->HFETAsourcePrimeDrainPrimePtr+1) -= xds;
        }
    }
    return(OK);
}
