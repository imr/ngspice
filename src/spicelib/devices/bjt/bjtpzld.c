/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"


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
            gx= *(ckt->CKTstate0 + here->BJTgx);
            xcpi= *(ckt->CKTstate0 + here->BJTcqbe);
            xcmu= *(ckt->CKTstate0 + here->BJTcqbc);
            xcbx= *(ckt->CKTstate0 + here->BJTcqbx);
            xccs= *(ckt->CKTstate0 + here->BJTcqcs);
            xcmcb= *(ckt->CKTstate0 + here->BJTcexbc);
            
	    *(here->BJTcolColPtr) +=                 m * (gcpr);
            *(here->BJTbaseBasePtr) +=               m * ((gx) + (xcbx) * (s->real));
            *(here->BJTbaseBasePtr + 1) +=           m *  ((xcbx) * (s->imag));
            *(here->BJTemitEmitPtr) +=               m *  (gepr);
            *(here->BJTcolPrimeColPrimePtr) +=       m * ((gmu+go+gcpr)
                                        + (xcmu+xccs+xcbx) * (s->real));
            *(here->BJTcolPrimeColPrimePtr + 1) +=   m * ((xcmu+xccs+xcbx)
                                                 * (s->imag));
            *(here->BJTbasePrimeBasePrimePtr) +=     m * ((gx+gpi+gmu)
                                   + (xcpi+xcmu+xcmcb) * (s->real));
            *(here->BJTbasePrimeBasePrimePtr + 1) += m * ((xcpi+xcmu+xcmcb) 
                                                   * (s->imag));
            *(here->BJTemitPrimeEmitPrimePtr) +=     m * ((gpi+gepr+gm+go)
                                           + (xcpi+xgm) * (s->real));
            *(here->BJTemitPrimeEmitPrimePtr + 1) += m * ((xcpi+xgm)
                                                 * (s->imag));
            *(here->BJTcolColPrimePtr) +=            m * (-gcpr);
            *(here->BJTbaseBasePrimePtr) +=          m * (-gx);
            *(here->BJTemitEmitPrimePtr) +=          m * (-gepr);
            *(here->BJTcolPrimeColPtr) +=            m * (-gcpr);
            *(here->BJTcolPrimeBasePrimePtr) +=      m * ((-gmu+gm)
                                           +   (-xcmu+xgm) * (s->real));
            *(here->BJTcolPrimeBasePrimePtr + 1) +=  m * ((-xcmu+xgm)
                                                * (s->imag));
            *(here->BJTcolPrimeEmitPrimePtr) +=      m * ((-gm-go)
                                             + (-xgm) * (s->real));
            *(here->BJTcolPrimeEmitPrimePtr + 1) +=  m * ((-xgm) *
                                                  (s->imag));
            *(here->BJTbasePrimeBasePtr) +=          m * (-gx);
            *(here->BJTbasePrimeColPrimePtr) +=      m * ((-gmu)
                                        +   (-xcmu-xcmcb) * (s->real));
            *(here->BJTbasePrimeColPrimePtr + 1) +=  m * ((-xcmu-xcmcb)
                                                * (s->imag));
            *(here->BJTbasePrimeEmitPrimePtr) +=     m * ((-gpi)
                                              +  (-xcpi) * (s->real));
            *(here->BJTbasePrimeEmitPrimePtr + 1) += m * ((-xcpi) 
                                                 * (s->imag));
            *(here->BJTemitPrimeEmitPtr) +=          m * (-gepr);
            *(here->BJTemitPrimeColPrimePtr) +=      m * ((-go)
                                              +   (xcmcb) * (s->real));
            *(here->BJTemitPrimeColPrimePtr + 1) +=  m * ((xcmcb) 
                                               *  (s->imag));
            *(here->BJTemitPrimeBasePrimePtr) +=     m * ((-gpi-gm)
                                      +  (-xcpi-xgm-xcmcb) * (s->real));
            *(here->BJTemitPrimeBasePrimePtr + 1) += m * ((-xcpi-xgm-xcmcb)
                             * (s->imag));
            *(here->BJTsubstSubstPtr) +=             m * ((xccs) * (s->real));
            *(here->BJTsubstSubstPtr + 1) +=         m * ((xccs) * (s->imag));
            *(here->BJTcolPrimeSubstPtr) +=          m * ((-xccs) * (s->real));
            *(here->BJTcolPrimeSubstPtr + 1) +=      m * ((-xccs) * (s->imag));
            *(here->BJTsubstColPrimePtr) +=          m * ((-xccs) * (s->real));
            *(here->BJTsubstColPrimePtr + 1) +=      m * ((-xccs) * (s->imag));
            *(here->BJTbaseColPrimePtr) +=           m * ((-xcbx) * (s->real));
            *(here->BJTbaseColPrimePtr + 1) +=       m * ((-xcbx) * (s->imag));
            *(here->BJTcolPrimeBasePtr) +=           m * ((-xcbx) * (s->real));
            *(here->BJTcolPrimeBasePtr + 1) +=       m * ((-xcbx) * (s->imag));
        }
    }
    return(OK);
}
