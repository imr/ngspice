/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * Function to load the COMPLEX circuit matrix using the
 * small signal parameters saved during a previous DC operating
 * point analysis.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BJTacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    BJTinstance *here;
    BJTmodel *model = (BJTmodel*)inModel;
    double gcpr;
    double gepr;
    double gpi;
    double gmu;
    double go;
    double xgm;
    double td;
    double arg;
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
            td=model->BJTexcessPhaseFactor;
            if(td != 0) {
                arg = td*ckt->CKTomega;
                gm = gm+go;
                xgm = -gm * sin(arg);
                gm = gm * cos(arg)-go;
            }
            gx= *(ckt->CKTstate0 + here->BJTgx);
            xcpi= *(ckt->CKTstate0 + here->BJTcqbe) * ckt->CKTomega;
            xcmu= *(ckt->CKTstate0 + here->BJTcqbc) * ckt->CKTomega;
            xcbx= *(ckt->CKTstate0 + here->BJTcqbx) * ckt->CKTomega;
            xcsub= *(ckt->CKTstate0 + here->BJTcqsub) * ckt->CKTomega;
            xcmcb= *(ckt->CKTstate0 + here->BJTcexbc) * ckt->CKTomega;
            xcbcx= *(ckt->CKTstate0 + here->BJTcqbcx) * ckt->CKTomega;

            *(here->BJTcolColPtr) +=                   m * (gcpr);
            *(here->BJTbaseBasePtr) +=                 m * (gx);
            *(here->BJTbaseBasePtr + 1) +=             m * (xcbx);
            *(here->BJTemitEmitPtr) +=                 m * (gepr);
            *(here->BJTcolPrimeColPrimePtr) +=         m * (gmu+go);
            *(here->BJTcollCXcollCXPtr) +=             m * (gcpr);
            *(here->BJTcolPrimeColPrimePtr + 1) +=     m * (xcmu+xcbx);
            *(here->BJTsubstConSubstConPtr + 1) +=     m * (xcsub);
            *(here->BJTbasePrimeBasePrimePtr) +=       m * (gx+gpi+gmu);
            *(here->BJTbasePrimeBasePrimePtr + 1) +=   m * (xcpi+xcmu+xcmcb);
            *(here->BJTemitPrimeEmitPrimePtr) +=       m * (gpi+gepr+gm+go);
            *(here->BJTemitPrimeEmitPrimePtr + 1) +=   m * (xcpi+xgm);
            *(here->BJTcollCollCXPtr) +=               m * (-gcpr);
            *(here->BJTbaseBasePrimePtr) +=            m * (-gx);
            *(here->BJTemitEmitPrimePtr) +=            m * (-gepr);
            *(here->BJTcollCXCollPtr) +=               m * (-gcpr);
            *(here->BJTcolPrimeBasePrimePtr) +=        m * (-gmu+gm);
            *(here->BJTcolPrimeBasePrimePtr + 1) +=    m * (-xcmu+xgm);
            *(here->BJTcolPrimeEmitPrimePtr) +=        m * (-gm-go);
            *(here->BJTcolPrimeEmitPrimePtr + 1) +=    m * (-xgm);
            *(here->BJTbasePrimeBasePtr) +=            m * (-gx);
            *(here->BJTbasePrimeColPrimePtr) +=        m * (-gmu);
            *(here->BJTbasePrimeColPrimePtr + 1) +=    m * (-xcmu-xcmcb);
            *(here->BJTbasePrimeEmitPrimePtr) +=       m * (-gpi);
            *(here->BJTbasePrimeEmitPrimePtr + 1) +=   m * (-xcpi);
            *(here->BJTemitPrimeEmitPtr) +=            m * (-gepr);
            *(here->BJTemitPrimeColPrimePtr) +=        m * (-go);
            *(here->BJTemitPrimeColPrimePtr + 1) +=    m * (xcmcb);
            *(here->BJTemitPrimeBasePrimePtr) +=       m * (-gpi-gm);
            *(here->BJTemitPrimeBasePrimePtr + 1) +=   m * (-xcpi-xgm-xcmcb);
            *(here->BJTsubstSubstPtr + 1) +=           m * (xcsub);
            *(here->BJTsubstConSubstPtr + 1) +=        m * (-xcsub);
            *(here->BJTsubstSubstConPtr + 1) +=        m * (-xcsub);
            *(here->BJTbaseColPrimePtr + 1) +=         m * (-xcbx);
            *(here->BJTcolPrimeBasePtr + 1) +=         m * (-xcbx);
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
                *(here->BJTbasePrimeBasePrimePtr + 1) += m *  xcbcx;
                *(here->BJTcollCXcollCXPtr + 1)       += m *  xcbcx;
                *(here->BJTbasePrimeCollCXPtr + 1)    += m * -xcbcx;
                *(here->BJTcollCXBasePrimePtr + 1)    += m * -xcbcx;
            }
        }
    }
    return(OK);
}
