/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

/*
 * Function to load the COMPLEX circuit matrix using the
 * small signal parameters saved during a previous DC operating
 * point analysis.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"

int
BJT2acLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;

{
    BJT2instance *here;
    BJT2model *model = (BJT2model*)inModel;
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

    for( ; model != NULL; model = model->BJT2nextModel) {
        for( here = model->BJT2instances; here!= NULL; 
                here = here->BJT2nextInstance) {
            

            gcpr=model->BJT2collectorConduct * here->BJT2area;
            gepr=model->BJT2emitterConduct * here->BJT2area;
            gpi= *(ckt->CKTstate0 + here->BJT2gpi);
            gmu= *(ckt->CKTstate0 + here->BJT2gmu);
            gm= *(ckt->CKTstate0 + here->BJT2gm);
            go= *(ckt->CKTstate0 + here->BJT2go);
            xgm=0;
            td=model->BJT2excessPhaseFactor;
            if(td != 0) {
                arg = td*ckt->CKTomega;
                gm = gm+go;
                xgm = -gm * sin(arg);
                gm = gm * cos(arg)-go;
            }
            gx= *(ckt->CKTstate0 + here->BJT2gx);
            xcpi= *(ckt->CKTstate0 + here->BJT2cqbe) * ckt->CKTomega;
            xcmu= *(ckt->CKTstate0 + here->BJT2cqbc) * ckt->CKTomega;
            xcbx= *(ckt->CKTstate0 + here->BJT2cqbx) * ckt->CKTomega;
            xcsub= *(ckt->CKTstate0 + here->BJT2cqsub) * ckt->CKTomega;
            xcmcb= *(ckt->CKTstate0 + here->BJT2cexbc) * ckt->CKTomega;
            *(here->BJT2colColPtr) +=   (gcpr);
            *(here->BJT2baseBasePtr) +=   (gx);
            *(here->BJT2baseBasePtr + 1) +=   (xcbx);
            *(here->BJT2emitEmitPtr) +=   (gepr);
            *(here->BJT2colPrimeColPrimePtr) +=   (gmu+go+gcpr);
            *(here->BJT2colPrimeColPrimePtr + 1) +=   (xcmu+xcbx);
            *(here->BJT2substConSubstConPtr + 1) +=   (xcsub);
            *(here->BJT2basePrimeBasePrimePtr) +=   (gx+gpi+gmu);
            *(here->BJT2basePrimeBasePrimePtr + 1) +=   (xcpi+xcmu+xcmcb);
            *(here->BJT2emitPrimeEmitPrimePtr) +=   (gpi+gepr+gm+go);
            *(here->BJT2emitPrimeEmitPrimePtr + 1) +=   (xcpi+xgm);
            *(here->BJT2colColPrimePtr) +=   (-gcpr);
            *(here->BJT2baseBasePrimePtr) +=   (-gx);
            *(here->BJT2emitEmitPrimePtr) +=   (-gepr);
            *(here->BJT2colPrimeColPtr) +=   (-gcpr);
            *(here->BJT2colPrimeBasePrimePtr) +=   (-gmu+gm);
            *(here->BJT2colPrimeBasePrimePtr + 1) +=   (-xcmu+xgm);
            *(here->BJT2colPrimeEmitPrimePtr) +=   (-gm-go);
            *(here->BJT2colPrimeEmitPrimePtr + 1) +=   (-xgm);
            *(here->BJT2basePrimeBasePtr) +=   (-gx);
            *(here->BJT2basePrimeColPrimePtr) +=   (-gmu);
            *(here->BJT2basePrimeColPrimePtr + 1) +=   (-xcmu-xcmcb);
            *(here->BJT2basePrimeEmitPrimePtr) +=   (-gpi);
            *(here->BJT2basePrimeEmitPrimePtr + 1) +=   (-xcpi);
            *(here->BJT2emitPrimeEmitPtr) +=   (-gepr);
            *(here->BJT2emitPrimeColPrimePtr) +=   (-go);
            *(here->BJT2emitPrimeColPrimePtr + 1) +=   (xcmcb);
            *(here->BJT2emitPrimeBasePrimePtr) +=   (-gpi-gm);
            *(here->BJT2emitPrimeBasePrimePtr + 1) +=   (-xcpi-xgm-xcmcb);
            *(here->BJT2substSubstPtr + 1) +=   (xcsub);
            *(here->BJT2substConSubstPtr + 1) +=   (-xcsub);
            *(here->BJT2substSubstConPtr + 1) +=   (-xcsub);
            *(here->BJT2baseColPrimePtr + 1) +=   (-xcbx);
            *(here->BJT2colPrimeBasePtr + 1) +=   (-xcbx);
        }
    }
    return(OK);
}
