/**********
Based on jfetacld.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994:  New call to PSacload() with matrix loading
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/sperror.h"
#include "psmodel.h"
#include "ngspice/suffix.h"

int
JFET2acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance *here;
    double gdpr;
    double gspr;
    double gm;
    double gds;
    double ggs;
    double xgs;
    double ggd;
    double xgd;
    double xgm, xgds, vgd, vgs, cd;

    double m; 

    for( ; model != NULL; model = JFET2nextModel(model)) {
        
        for( here = JFET2instances(model); here != NULL; 
                here = JFET2nextInstance(here)) {

            gdpr=model->JFET2drainConduct * here->JFET2area;
            gspr=model->JFET2sourceConduct * here->JFET2area;
            gm= *(ckt->CKTstate0 + here->JFET2gm) ;
            gds= *(ckt->CKTstate0 + here->JFET2gds) ;
            ggs= *(ckt->CKTstate0 + here->JFET2ggs) ;
            xgs= *(ckt->CKTstate0 + here->JFET2qgs) * ckt->CKTomega ;
            ggd= *(ckt->CKTstate0 + here->JFET2ggd) ;
            xgd= *(ckt->CKTstate0 + here->JFET2qgd) * ckt->CKTomega ;

            vgs = *(ckt->CKTstate0 + here->JFET2vgs);
            vgd = *(ckt->CKTstate0 + here->JFET2vgd);
            cd = *(ckt->CKTstate0 + here->JFET2cd);
            PSacload(ckt,model, here, vgs, vgd, cd, ckt->CKTomega,
                                      &gm, &xgm, &gds, &xgds);
            xgds += *(ckt->CKTstate0 + here->JFET2qds) * ckt->CKTomega ;

            m = here->JFET2m;

            *(here->JFET2drainPrimeDrainPrimePtr +1)   += m * (xgds);
            *(here->JFET2sourcePrimeSourcePrimePtr +1) += m * (xgds+xgm);
            *(here->JFET2drainPrimeGatePtr +1)         += m *      (xgm);
            *(here->JFET2drainPrimeSourcePrimePtr +1)  -= m * (xgds+xgm);
            *(here->JFET2sourcePrimeGatePtr +1)        -= m *      (xgm);
            *(here->JFET2sourcePrimeDrainPrimePtr +1)  -= m * (xgds);

            *(here->JFET2drainDrainPtr )               += m * (gdpr);
            *(here->JFET2gateGatePtr )                 += m * (ggd+ggs);
            *(here->JFET2gateGatePtr +1)               += m * (xgd+xgs);
            *(here->JFET2sourceSourcePtr )             += m * (gspr);
            *(here->JFET2drainPrimeDrainPrimePtr )     += m * (gdpr+gds+ggd);
            *(here->JFET2drainPrimeDrainPrimePtr +1)   += m * (xgd);
            *(here->JFET2sourcePrimeSourcePrimePtr )   += m * (gspr+gds+gm+ggs);
            *(here->JFET2sourcePrimeSourcePrimePtr +1) += m * (xgs);
            *(here->JFET2drainDrainPrimePtr )          -= m * (gdpr);
            *(here->JFET2gateDrainPrimePtr )           -= m * (ggd);
            *(here->JFET2gateDrainPrimePtr +1)         -= m * (xgd);
            *(here->JFET2gateSourcePrimePtr )          -= m * (ggs);
            *(here->JFET2gateSourcePrimePtr +1)        -= m * (xgs);
            *(here->JFET2sourceSourcePrimePtr )        -= m * (gspr);
            *(here->JFET2drainPrimeDrainPtr )          -= m * (gdpr);
            *(here->JFET2drainPrimeGatePtr )           += m * (-ggd+gm);
            *(here->JFET2drainPrimeGatePtr +1)         -= m * (xgd);
            *(here->JFET2drainPrimeSourcePrimePtr )    += m * (-gds-gm);
            *(here->JFET2sourcePrimeGatePtr )          += m * (-ggs-gm);
            *(here->JFET2sourcePrimeGatePtr +1)        -= m * (xgs);
            *(here->JFET2sourcePrimeSourcePtr )        -= m * (gspr);
            *(here->JFET2sourcePrimeDrainPrimePtr )    -= m * (gds);

        }
    }
    return(OK);
}
