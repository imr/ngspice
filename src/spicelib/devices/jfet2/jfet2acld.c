/**********
Based on jfetacld.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994:  New call to PSacload() with matrix loading
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "jfet2defs.h"
#include "sperror.h"
#include "psmodel.h"
#include "suffix.h"

int
JFET2acLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
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

    for( ; model != NULL; model = model->JFET2nextModel ) {
        
        for( here = model->JFET2instances; here != NULL; 
                here = here->JFET2nextInstance) {
            if (here->JFET2owner != ARCHme) continue;

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
            *(here->JFET2drainPrimeDrainPrimePtr +1)   += xgds;
            *(here->JFET2sourcePrimeSourcePrimePtr +1) += xgds+xgm;
            *(here->JFET2drainPrimeGatePtr +1)         +=      xgm;
            *(here->JFET2drainPrimeSourcePrimePtr +1)  -= xgds+xgm;
            *(here->JFET2sourcePrimeGatePtr +1)        -=      xgm;
            *(here->JFET2sourcePrimeDrainPrimePtr +1)  -= xgds;

            *(here->JFET2drainDrainPtr ) += gdpr;
            *(here->JFET2gateGatePtr ) += ggd+ggs;
            *(here->JFET2gateGatePtr +1) += xgd+xgs;
            *(here->JFET2sourceSourcePtr ) += gspr;
            *(here->JFET2drainPrimeDrainPrimePtr ) += gdpr+gds+ggd;
            *(here->JFET2drainPrimeDrainPrimePtr +1) += xgd;
            *(here->JFET2sourcePrimeSourcePrimePtr ) += gspr+gds+gm+ggs;
            *(here->JFET2sourcePrimeSourcePrimePtr +1) += xgs;
            *(here->JFET2drainDrainPrimePtr ) -= gdpr;
            *(here->JFET2gateDrainPrimePtr ) -= ggd;
            *(here->JFET2gateDrainPrimePtr +1) -= xgd;
            *(here->JFET2gateSourcePrimePtr ) -= ggs;
            *(here->JFET2gateSourcePrimePtr +1) -= xgs;
            *(here->JFET2sourceSourcePrimePtr ) -= gspr;
            *(here->JFET2drainPrimeDrainPtr ) -= gdpr;
            *(here->JFET2drainPrimeGatePtr ) += (-ggd+gm);
            *(here->JFET2drainPrimeGatePtr +1) -= xgd;
            *(here->JFET2drainPrimeSourcePrimePtr ) += (-gds-gm);
            *(here->JFET2sourcePrimeGatePtr ) += (-ggs-gm);
            *(here->JFET2sourcePrimeGatePtr +1) -= xgs;
            *(here->JFET2sourcePrimeSourcePtr ) -= gspr;
            *(here->JFET2sourcePrimeDrainPrimePtr ) -= gds;

        }
    }
    return(OK);
}
