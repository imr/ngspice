/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Dietmar Warning 2003
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    double gspr, gsprsw;
    double geq;
    double xceq;
    DIOinstance *here;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {
            gspr=here->DIOtConductance;
            geq= *(ckt->CKTstate0 + here->DIOconduct);
            xceq= *(ckt->CKTstate0 + here->DIOcapCurrent) * ckt->CKTomega;
            *(here->DIOposPosPtr ) += gspr;
            *(here->DIOnegNegPtr ) += geq;
            *(here->DIOnegNegPtr +1 ) += xceq;
            *(here->DIOposPrimePosPrimePtr ) += geq+gspr;
            *(here->DIOposPrimePosPrimePtr +1 ) += xceq;
            *(here->DIOposPosPrimePtr ) -= gspr;
            *(here->DIOnegPosPrimePtr ) -= geq;
            *(here->DIOnegPosPrimePtr +1 ) -= xceq;
            *(here->DIOposPrimePosPtr ) -= gspr;
            *(here->DIOposPrimeNegPtr ) -= geq;
            *(here->DIOposPrimeNegPtr +1 ) -= xceq;
            if (model->DIOresistSWGiven) {
                gsprsw=here->DIOtConductanceSW;
                geq= *(ckt->CKTstate0 + here->DIOconductSW);
                xceq= *(ckt->CKTstate0 + here->DIOcapCurrentSW) * ckt->CKTomega;
                *(here->DIOposPosPtr) += gsprsw;
                *(here->DIOnegNegPtr) += geq;
                *(here->DIOnegNegPtr + 1) += xceq;
                *(here->DIOposSwPrimePosSwPrimePtr) += (geq + gsprsw);
                *(here->DIOposSwPrimePosSwPrimePtr + 1) += xceq;
                *(here->DIOposPosSwPrimePtr) -= gsprsw;
                *(here->DIOnegPosSwPrimePtr) -= geq;
                *(here->DIOnegPosSwPrimePtr + 1) -= xceq;
                *(here->DIOposSwPrimePosPtr) -= gsprsw;
                *(here->DIOposSwPrimeNegPtr) -= geq;
                *(here->DIOposSwPrimeNegPtr + 1) -= xceq;
            }

            int selfheat = ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given));
            if (selfheat) {
                double dIth_dVrs = here->DIOdIth_dVrs;
                double dIth_dVdio = here->DIOdIth_dVdio;
                double dIth_dT = here->DIOdIth_dT;
                double gcTt = here->DIOgcTt;
                double dIrs_dT = here->DIOdIrs_dT;
                double dIdio_dT = *(ckt->CKTstate0 + here->DIOdIdio_dT);
                (*(here->DIOtempPosPtr)      += -dIth_dVrs);
                (*(here->DIOtempPosPrimePtr) += -dIth_dVdio + dIth_dVrs);
                (*(here->DIOtempNegPtr)      +=  dIth_dVdio);
                (*(here->DIOtempTempPtr)     += -dIth_dT + 1/model->DIOrth0 + gcTt);
                (*(here->DIOposTempPtr)      +=  dIrs_dT);
                (*(here->DIOposPrimeTempPtr) +=  dIdio_dT - dIrs_dT);
                (*(here->DIOnegTempPtr)      += -dIdio_dT);

                double xgcTt= *(ckt->CKTstate0 + here->DIOcqth) * ckt->CKTomega;
                (*(here->DIOtempTempPtr + 1) +=  xgcTt);

                if (model->DIOresistSWGiven) {
                    double dIth_dVrssw = here->DIOdIth_dVrs;
                    double dIth_dVdioSw = here->DIOdIth_dVdio;
                    double dIrssw_dT = here->DIOdIrs_dT;
                    double dIdioSw_dT = *(ckt->CKTstate0 + here->DIOdIdio_dT);
                    (*(here->DIOtempPosPtr)        += -dIth_dVrssw);
                    (*(here->DIOtempPosSwPrimePtr) += -dIth_dVdioSw + dIth_dVrssw);
                    (*(here->DIOtempNegPtr)        +=  dIth_dVdioSw);
                    (*(here->DIOposTempPtr)        +=  dIrssw_dT);
                    (*(here->DIOposSwPrimeTempPtr) +=  dIdioSw_dT - dIrssw_dT);
                    (*(here->DIOnegTempPtr)        += -dIdioSw_dT);
                }
            }
        }
    }
    return(OK);

}
