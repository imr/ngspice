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
BJT2acLoad(GENmodel *inModel, CKTcircuit *ckt)
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
    double m;

    for( ; model != NULL; model = model->BJT2nextModel) {
        for( here = model->BJT2instances; here!= NULL; 
                here = here->BJT2nextInstance) {
            
	    if (here->BJT2owner != ARCHme) continue;

            m = here->BJT2m;
	    
            gcpr=here->BJT2tCollectorConduct * here->BJT2area;
            gepr=here->BJT2tEmitterConduct * here->BJT2area;
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
            *(here->BJT2colColPtr) +=                   m * (gcpr);
            *(here->BJT2baseBasePtr) +=                 m * (gx);
            *(here->BJT2baseBasePtr + 1) +=             m * (xcbx);
            *(here->BJT2emitEmitPtr) +=                 m * (gepr);
            *(here->BJT2colPrimeColPrimePtr) +=         m * (gmu+go+gcpr);
            *(here->BJT2colPrimeColPrimePtr + 1) +=     m * (xcmu+xcbx);
            *(here->BJT2substConSubstConPtr + 1) +=     m * (xcsub);
            *(here->BJT2basePrimeBasePrimePtr) +=       m * (gx+gpi+gmu);
            *(here->BJT2basePrimeBasePrimePtr + 1) +=   m * (xcpi+xcmu+xcmcb);
            *(here->BJT2emitPrimeEmitPrimePtr) +=       m * (gpi+gepr+gm+go);
            *(here->BJT2emitPrimeEmitPrimePtr + 1) +=   m * (xcpi+xgm);
            *(here->BJT2colColPrimePtr) +=              m * (-gcpr);
            *(here->BJT2baseBasePrimePtr) +=            m * (-gx);
            *(here->BJT2emitEmitPrimePtr) +=            m * (-gepr);
            *(here->BJT2colPrimeColPtr) +=              m * (-gcpr);
            *(here->BJT2colPrimeBasePrimePtr) +=        m * (-gmu+gm);
            *(here->BJT2colPrimeBasePrimePtr + 1) +=    m * (-xcmu+xgm);
            *(here->BJT2colPrimeEmitPrimePtr) +=        m * (-gm-go);
            *(here->BJT2colPrimeEmitPrimePtr + 1) +=    m * (-xgm);
            *(here->BJT2basePrimeBasePtr) +=            m * (-gx);
            *(here->BJT2basePrimeColPrimePtr) +=        m * (-gmu);
            *(here->BJT2basePrimeColPrimePtr + 1) +=    m * (-xcmu-xcmcb);
            *(here->BJT2basePrimeEmitPrimePtr) +=       m * (-gpi);
            *(here->BJT2basePrimeEmitPrimePtr + 1) +=   m * (-xcpi);
            *(here->BJT2emitPrimeEmitPtr) +=            m * (-gepr);
            *(here->BJT2emitPrimeColPrimePtr) +=        m * (-go);
            *(here->BJT2emitPrimeColPrimePtr + 1) +=    m * (xcmcb);
            *(here->BJT2emitPrimeBasePrimePtr) +=       m * (-gpi-gm);
            *(here->BJT2emitPrimeBasePrimePtr + 1) +=   m * (-xcpi-xgm-xcmcb);
            *(here->BJT2substSubstPtr + 1) +=           m * (xcsub);
            *(here->BJT2substConSubstPtr + 1) +=        m * (-xcsub);
            *(here->BJT2substSubstConPtr + 1) +=        m * (-xcsub);
            *(here->BJT2baseColPrimePtr + 1) +=         m * (-xcbx);
            *(here->BJT2colPrimeBasePtr + 1) +=         m * (-xcbx);
        }
    }
    return(OK);
}
