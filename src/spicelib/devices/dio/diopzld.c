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
                xceq= *(ckt->CKTstate0 + here->DIOcapCurrentSW) * ckt->CKTomega;
                *(here->DIOposPosPtr) += gsprsw;
                *(here->DIOnegNegPtr) += geq + xceq * s->real;
                *(here->DIOnegNegPtr + 1) += xceq * s->imag;
                *(here->DIOposSwPrimePosSwPrimePtr) += geq + gsprsw + xceq * s->real;
                *(here->DIOposSwPrimePosSwPrimePtr + 1) += xceq * s->imag;
                *(here->DIOposPosSwPrimePtr) -= gsprsw;
                *(here->DIOnegPosSwPrimePtr) -= geq;
                *(here->DIOnegPosSwPrimePtr + 1) -= xceq;
                *(here->DIOposSwPrimePosPtr) -= gsprsw;
                *(here->DIOposSwPrimeNegPtr) -= geq + xceq * s->real;
                *(here->DIOposSwPrimeNegPtr + 1) -= xceq * s->imag;
            }
        }
    }
    return(OK);

}
