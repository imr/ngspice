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
#include <stdio.h>
#include "cktdefs.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"

int
BJTacLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;

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

    for( ; model != NULL; model = model->BJTnextModel) {
        for( here = model->BJTinstances; here!= NULL; 
                here = here->BJTnextInstance) {
	    if (here->BJTowner != ARCHme) continue;

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
            *(here->BJTcolColPtr) +=   (gcpr);
            *(here->BJTbaseBasePtr) +=   (gx);
            *(here->BJTbaseBasePtr + 1) +=   (xcbx);
            *(here->BJTemitEmitPtr) +=   (gepr);
            *(here->BJTcolPrimeColPrimePtr) +=   (gmu+go+gcpr);
            *(here->BJTcolPrimeColPrimePtr + 1) +=   (xcmu+xccs+xcbx);
            *(here->BJTbasePrimeBasePrimePtr) +=   (gx+gpi+gmu);
            *(here->BJTbasePrimeBasePrimePtr + 1) +=   (xcpi+xcmu+xcmcb);
            *(here->BJTemitPrimeEmitPrimePtr) +=   (gpi+gepr+gm+go);
            *(here->BJTemitPrimeEmitPrimePtr + 1) +=   (xcpi+xgm);
            *(here->BJTcolColPrimePtr) +=   (-gcpr);
            *(here->BJTbaseBasePrimePtr) +=   (-gx);
            *(here->BJTemitEmitPrimePtr) +=   (-gepr);
            *(here->BJTcolPrimeColPtr) +=   (-gcpr);
            *(here->BJTcolPrimeBasePrimePtr) +=   (-gmu+gm);
            *(here->BJTcolPrimeBasePrimePtr + 1) +=   (-xcmu+xgm);
            *(here->BJTcolPrimeEmitPrimePtr) +=   (-gm-go);
            *(here->BJTcolPrimeEmitPrimePtr + 1) +=   (-xgm);
            *(here->BJTbasePrimeBasePtr) +=   (-gx);
            *(here->BJTbasePrimeColPrimePtr) +=   (-gmu);
            *(here->BJTbasePrimeColPrimePtr + 1) +=   (-xcmu-xcmcb);
            *(here->BJTbasePrimeEmitPrimePtr) +=   (-gpi);
            *(here->BJTbasePrimeEmitPrimePtr + 1) +=   (-xcpi);
            *(here->BJTemitPrimeEmitPtr) +=   (-gepr);
            *(here->BJTemitPrimeColPrimePtr) +=   (-go);
            *(here->BJTemitPrimeColPrimePtr + 1) +=   (xcmcb);
            *(here->BJTemitPrimeBasePrimePtr) +=   (-gpi-gm);
            *(here->BJTemitPrimeBasePrimePtr + 1) +=   (-xcpi-xgm-xcmcb);
            *(here->BJTsubstSubstPtr + 1) +=   (xccs);
            *(here->BJTcolPrimeSubstPtr + 1) +=   (-xccs);
            *(here->BJTsubstColPrimePtr + 1) +=   (-xccs);
            *(here->BJTbaseColPrimePtr + 1) +=   (-xcbx);
            *(here->BJTcolPrimeBasePtr + 1) +=   (-xcbx);
        }
    }
    return(OK);
}
