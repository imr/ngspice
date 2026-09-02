/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Dietmar Warning 2003
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "diodefs.h"
#include "ngspice/suffix.h"


int
DIOpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
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
            xceq= *(ckt->CKTstate0 + here->DIOcapCurrent);
            *(here->DIOposPosPtr ) += gspr;
            *(here->DIOnegNegPtr ) += geq + xceq * s->real;
            *(here->DIOnegNegPtr +1 ) += xceq * s->imag;
            *(here->DIOposPrimePosPrimePtr ) += geq + gspr + xceq * s->real;
            *(here->DIOposPrimePosPrimePtr +1 ) += xceq * s->imag;
            *(here->DIOposPosPrimePtr ) -= gspr;
            *(here->DIOnegPosPrimePtr ) -= geq + xceq * s->real;
            *(here->DIOnegPosPrimePtr +1 ) -= xceq * s->imag;
            *(here->DIOposPrimePosPtr ) -= gspr;
            *(here->DIOposPrimeNegPtr ) -= geq + xceq * s->real;
            *(here->DIOposPrimeNegPtr +1 ) -= xceq * s->imag;
            if (model->DIOresistSWGiven) {
                gsprsw=here->DIOtConductanceSW;
                geq= *(ckt->CKTstate0 + here->DIOconductSW);
                xceq= *(ckt->CKTstate0 + here->DIOcapCurrentSW);
                *(here->DIOposPosPtr) += gsprsw;
                *(here->DIOnegNegPtr) += geq + xceq * s->real;
                *(here->DIOnegNegPtr + 1) += xceq * s->imag;
                *(here->DIOposSwPrimePosSwPrimePtr) += geq + gsprsw + xceq * s->real;
                *(here->DIOposSwPrimePosSwPrimePtr + 1) += xceq * s->imag;
                *(here->DIOposPosSwPrimePtr) -= gsprsw;
                *(here->DIOnegPosSwPrimePtr) -= geq + xceq * s->real;
                *(here->DIOnegPosSwPrimePtr + 1) -= xceq * s->imag;
                *(here->DIOposSwPrimePosPtr) -= gsprsw;
                *(here->DIOposSwPrimeNegPtr) -= geq + xceq * s->real;
                *(here->DIOposSwPrimeNegPtr + 1) -= xceq * s->imag;
            }

            if (DIOselfheat(here)) {
                double dIth_dVrs = here->DIOdIth_dVrs;
                double dIth_dVdio = here->DIOdIth_dVdio;
                double dIth_dT = here->DIOdIth_dT;
                double dIrs_dT = here->DIOdIrs_dT;
                double dIdio_dT = *(ckt->CKTstate0 + here->DIOdIdio_dT);
                (*(here->DIOtempPosPtr )      += -dIth_dVrs);
                (*(here->DIOtempPosPrimePtr ) += -dIth_dVdio + dIth_dVrs);
                (*(here->DIOtempNegPtr )      +=  dIth_dVdio);
                /* thermal capacitance: cth0 is a model constant, the transient
                   companion values gcTt/DIOcqth have no meaning here */
                (*(here->DIOtempTempPtr )     += -dIth_dT + 1/model->DIOrth0
                                               + model->DIOcth0 * s->real);
                (*(here->DIOtempTempPtr +1 )  +=  model->DIOcth0 * s->imag);
                (*(here->DIOposTempPtr )      +=  dIrs_dT);
                (*(here->DIOposPrimeTempPtr ) +=  dIdio_dT - dIrs_dT);
                (*(here->DIOnegTempPtr )      += -dIdio_dT);

                if (model->DIOresistSWGiven) {
                    /* dIth_dT already holds bottom + sidewall, do not add again */
                    double dIth_dVrssw = here->DIOdIth_dVrssw;
                    double dIth_dVdioSw = here->DIOdIth_dVdioSw;
                    double dIrssw_dT = here->DIOdIrssw_dT;
                    double dIdioSw_dT = *(ckt->CKTstate0 + here->DIOdIdioSW_dT);
                    (*(here->DIOtempPosPtr )        += -dIth_dVrssw);
                    (*(here->DIOtempPosSwPrimePtr ) += -dIth_dVdioSw + dIth_dVrssw);
                    (*(here->DIOtempNegPtr )        +=  dIth_dVdioSw);
                    (*(here->DIOposTempPtr )        +=  dIrssw_dT);
                    (*(here->DIOposSwPrimeTempPtr ) +=  dIdioSw_dT - dIrssw_dT);
                    (*(here->DIOnegTempPtr )        += -dIdioSw_dT);
                }
            }

            if (DIOrevrec(here)) {
                /* QP subcircuit */
                double gdres = *(ckt->CKTstate0 + here->DIOresConduct);
                double dcrrdvd = here->DIOcurFactor * gdres;
                double capsr = here->DIOtTransitTime;
                double gain0 = here->DIOqpGainScaled * capsr;
                *(here->DIOqpQpPtr )         += 1/model->DIOsoftRevRecParam
                                              + capsr * s->real;
                *(here->DIOqpQpPtr +1 )      += capsr * s->imag;
                *(here->DIOqpPosPrimePtr )   += -dcrrdvd;
                *(here->DIOqpNegPtr )        +=  dcrrdvd;
                /* PZ: gqcsr -> s*capsr, capsr = TT */
                *(here->DIOposPrimeQpPtr )   +=  gain0 * s->real;
                *(here->DIOposPrimeQpPtr +1 ) +=  gain0 * s->imag;
                *(here->DIOnegQpPtr )        += -gain0 * s->real;
                *(here->DIOnegQpPtr +1 )     += -gain0 * s->imag;
                if (DIOselfheat(here)) {
                    double vdop = *(ckt->CKTstate0 + here->DIOvoltage);
                    *(here->DIOqpTempPtr )   += -here->DIOcurFactor
                                              * *(ckt->CKTstate0 + here->DIOdIdio_dT);
                    *(here->DIOtempQpPtr )    += -vdop * gain0 * s->real;
                    *(here->DIOtempQpPtr +1 ) += -vdop * gain0 * s->imag;
                }
            }

        }
    }
    return(OK);

}
