/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BJTpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;
    double gcpr;
    double gepr;
    double gpi;
    double gmu;
    double go;
    double xgm;
    double gm;
    double gx;
    double xcpi;
    double xcmu;
    double xcbx;
    double xcsub;
    double xcmcb;
    double m;
    double Irci_Vrci, Irci_Vbci, Irci_Vbcx, xcbcx;

    for( ; model != NULL; model = BJTnextModel(model)) {
        for( here = BJTinstances(model); here!= NULL;
                here = BJTnextInstance(here)) {

            m = here->BJTm;

            gcpr=here->BJTtcollectorConduct;
            gepr=here->BJTtemitterConduct;
            gpi= *(ckt->CKTstate0 + here->BJTgpi);
            gmu= *(ckt->CKTstate0 + here->BJTgmu);
            gm= *(ckt->CKTstate0 + here->BJTgm);
            go= *(ckt->CKTstate0 + here->BJTgo);
            Irci_Vrci = *(ckt->CKTstate0 + here->BJTirci_Vrci);
            Irci_Vbci = *(ckt->CKTstate0 + here->BJTirci_Vbci);
            Irci_Vbcx = *(ckt->CKTstate0 + here->BJTirci_Vbcx);
            xgm=0;
            gx= *(ckt->CKTstate0 + here->BJTgx);
            xcpi= *(ckt->CKTstate0 + here->BJTcqbe);
            xcmu= *(ckt->CKTstate0 + here->BJTcqbc);
            xcbx= *(ckt->CKTstate0 + here->BJTcqbx);
            xcsub= *(ckt->CKTstate0 + here->BJTcqsub);
            xcmcb= *(ckt->CKTstate0 + here->BJTcexbc);
            xcbcx= *(ckt->CKTstate0 + here->BJTcqbcx);

            *(here->BJTcolColPtr) +=                 m * (gcpr);
            *(here->BJTbaseBasePtr) +=               m * ((gx) + (xcbx) * (s->real));
            *(here->BJTbaseBasePtr + 1) +=           m * ((xcbx) * (s->imag));
            *(here->BJTemitEmitPtr) +=               m * (gepr);
            *(here->BJTcolPrimeColPrimePtr) +=       m * ((gmu+go) + (xcmu+xcbx) * (s->real));
            *(here->BJTcolPrimeColPrimePtr + 1) +=   m * ((xcmu+xcbx) * (s->imag));
            *(here->BJTcollCXcollCXPtr) +=           m * (gcpr);

            *(here->BJTsubstConSubstConPtr) +=       m * (xcsub)* (s->real);
            *(here->BJTsubstConSubstConPtr + 1) +=   m * (xcsub)* (s->imag);

            *(here->BJTbasePrimeBasePrimePtr) +=     m * ((gx+gpi+gmu) + (xcpi+xcmu+xcmcb) * (s->real));
            *(here->BJTbasePrimeBasePrimePtr + 1) += m * ((xcpi+xcmu+xcmcb) * (s->imag));
            *(here->BJTemitPrimeEmitPrimePtr) +=     m * ((gpi+gepr+gm+go) + (xcpi+xgm) * (s->real));
            *(here->BJTemitPrimeEmitPrimePtr + 1) += m * ((xcpi+xgm) * (s->imag));
            *(here->BJTcollCollCXPtr) +=             m * (-gcpr);
            *(here->BJTbaseBasePrimePtr) +=          m * (-gx);
            *(here->BJTemitEmitPrimePtr) +=          m * (-gepr);
            *(here->BJTcollCXCollPtr) +=             m * (-gcpr);
            *(here->BJTcolPrimeBasePrimePtr) +=      m * ((-gmu+gm) + (-xcmu+xgm) * (s->real));
            *(here->BJTcolPrimeBasePrimePtr + 1) +=  m * ((-xcmu+xgm) * (s->imag));
            *(here->BJTcolPrimeEmitPrimePtr) +=      m * ((-gm-go) + (-xgm) * (s->real));
            *(here->BJTcolPrimeEmitPrimePtr + 1) +=  m * ((-xgm) * (s->imag));
            *(here->BJTbasePrimeBasePtr) +=          m * (-gx);
            *(here->BJTbasePrimeColPrimePtr) +=      m * ((-gmu) +   (-xcmu-xcmcb) * (s->real));
            *(here->BJTbasePrimeColPrimePtr + 1) +=  m * ((-xcmu-xcmcb) * (s->imag));
            *(here->BJTbasePrimeEmitPrimePtr) +=     m * ((-gpi) + (-xcpi) * (s->real));
            *(here->BJTbasePrimeEmitPrimePtr + 1) += m * ((-xcpi) * (s->imag));
            *(here->BJTemitPrimeEmitPtr) +=          m * (-gepr);
            *(here->BJTemitPrimeColPrimePtr) +=      m * ((-go) + (xcmcb) * (s->real));
            *(here->BJTemitPrimeColPrimePtr + 1) +=  m * ((xcmcb) * (s->imag));
            *(here->BJTemitPrimeBasePrimePtr) +=     m * ((-gpi-gm) + (-xcpi-xgm-xcmcb) * (s->real));
            *(here->BJTemitPrimeBasePrimePtr + 1) += m * ((-xcpi-xgm-xcmcb) * (s->imag));
            *(here->BJTsubstSubstPtr) +=             m * ((xcsub) * (s->real));
            *(here->BJTsubstSubstPtr + 1) +=         m * ((xcsub) * (s->imag));
            *(here->BJTsubstConSubstPtr) +=          m * ((-xcsub) * (s->real));
            *(here->BJTsubstConSubstPtr + 1) +=      m * ((-xcsub) * (s->imag));
            *(here->BJTsubstSubstConPtr) +=          m * ((-xcsub) * (s->real));
            *(here->BJTsubstSubstConPtr + 1) +=      m * ((-xcsub) * (s->imag));
            *(here->BJTbaseColPrimePtr) +=           m * ((-xcbx) * (s->real));
            *(here->BJTbaseColPrimePtr + 1) +=       m * ((-xcbx) * (s->imag));
            *(here->BJTcolPrimeBasePtr) +=           m * ((-xcbx) * (s->real));
            *(here->BJTcolPrimeBasePtr + 1) +=       m * ((-xcbx) * (s->imag));
            if (model->BJTintCollResistGiven) {
                *(here->BJTcollCXcollCXPtr)    += m *  Irci_Vrci;
                *(here->BJTcollCXColPrimePtr)  += m * -Irci_Vrci;
                *(here->BJTcollCXBasePrimePtr) += m *  Irci_Vbci;
                *(here->BJTcollCXColPrimePtr)  += m * -Irci_Vbci;
                *(here->BJTcollCXBasePrimePtr) += m *  Irci_Vbcx;
                *(here->BJTcollCXcollCXPtr)    += m * -Irci_Vbcx;
                *(here->BJTcolPrimeCollCXPtr)    += m * -Irci_Vrci;
                *(here->BJTcolPrimeColPrimePtr)  += m *  Irci_Vrci;
                *(here->BJTcolPrimeBasePrimePtr) += m * -Irci_Vbci;
                *(here->BJTcolPrimeColPrimePtr)  += m *  Irci_Vbci;
                *(here->BJTcolPrimeBasePrimePtr) += m * -Irci_Vbcx;
                *(here->BJTcolPrimeCollCXPtr)    += m *  Irci_Vbcx;
                *(here->BJTbasePrimeBasePrimePtr)     += m *  xcbcx * (s->real);
                *(here->BJTbasePrimeBasePrimePtr + 1) += m *  xcbcx * (s->imag);
                *(here->BJTcollCXcollCXPtr)           += m *  xcbcx * (s->real);
                *(here->BJTcollCXcollCXPtr + 1)       += m *  xcbcx * (s->imag);
                *(here->BJTbasePrimeCollCXPtr)        += m * -xcbcx * (s->real);
                *(here->BJTbasePrimeCollCXPtr + 1)    += m * -xcbcx * (s->imag);
                *(here->BJTcollCXBasePrimePtr)        += m * -xcbcx * (s->real);
                *(here->BJTcollCXBasePrimePtr + 1)    += m * -xcbcx * (s->imag);
            }
        }
    }
    return(OK);
}
