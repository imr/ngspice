/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * Function to load the COMPLEX circuit matrix using the
 * small signal parameters saved during a previous DC operating
 * point analysis.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"

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
    double xccs;
    double xcmcb;
    double m;

    for( ; model != NULL; model = model->BJTnextModel) {
        for( here = model->BJTinstances; here!= NULL; 
                here = here->BJTnextInstance) {
	    if (here->BJTowner != ARCHme) continue;

            m = here->BJTm;
	    
            gcpr=model->BJTcollectorConduct * here->BJTarea;
            gepr=model->BJTemitterConduct * here->BJTarea;
            gpi= *(ckt->CKTstate0 + here->BJTgpi);
            gmu= *(ckt->CKTstate0 + here->BJTgmu);
            gm= *(ckt->CKTstate0 + here->BJTgm);
            go= *(ckt->CKTstate0 + here->BJTgo);
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
            xccs= *(ckt->CKTstate0 + here->BJTcqcs) * ckt->CKTomega;
            xcmcb= *(ckt->CKTstate0 + here->BJTcexbc) * ckt->CKTomega;
            *(here->BJTcolColPtr) +=                   m * (gcpr);
            *(here->BJTbaseBasePtr) +=                 m * (gx);
            *(here->BJTbaseBasePtr + 1) +=             m * (xcbx);
            *(here->BJTemitEmitPtr) +=                 m * (gepr);
            *(here->BJTcolPrimeColPrimePtr) +=         m * (gmu+go+gcpr);
            *(here->BJTcolPrimeColPrimePtr + 1) +=     m * (xcmu+xccs+xcbx);
            *(here->BJTbasePrimeBasePrimePtr) +=       m * (gx+gpi+gmu);
            *(here->BJTbasePrimeBasePrimePtr + 1) +=   m * (xcpi+xcmu+xcmcb);
            *(here->BJTemitPrimeEmitPrimePtr) +=       m * (gpi+gepr+gm+go);
            *(here->BJTemitPrimeEmitPrimePtr + 1) +=   m * (xcpi+xgm);
            *(here->BJTcolColPrimePtr) +=              m * (-gcpr);
            *(here->BJTbaseBasePrimePtr) +=            m * (-gx);
            *(here->BJTemitEmitPrimePtr) +=            m * (-gepr);
            *(here->BJTcolPrimeColPtr) +=              m * (-gcpr);
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
            *(here->BJTemitPrimeColPrimePtr + 1) +=    m *  (xcmcb);
            *(here->BJTemitPrimeBasePrimePtr) +=       m * (-gpi-gm);
            *(here->BJTemitPrimeBasePrimePtr + 1) +=   m * (-xcpi-xgm-xcmcb);
            *(here->BJTsubstSubstPtr + 1) +=           m * (xccs);
            *(here->BJTcolPrimeSubstPtr + 1) +=        m * (-xccs);
            *(here->BJTsubstColPrimePtr + 1) +=        m * (-xccs);
            *(here->BJTbaseColPrimePtr + 1) +=         m * (-xcbx);
            *(here->BJTcolPrimeBasePtr + 1) +=         m * (-xcbx);
        }
    }
    return(OK);
}
