/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


int
BJT2pzLoad(inModel,ckt,s)
    GENmodel *inModel;
    CKTcircuit *ckt;
    SPcomplex *s;

{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
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

    for( ; model != NULL; model = model->BJT2nextModel) {
        for( here = model->BJT2instances; here!= NULL; 
                here = here->BJT2nextInstance) {
            

            gcpr=model->BJT2collectorResist * here->BJT2area;
            gepr=model->BJT2emitterResist * here->BJT2area;
            gpi= *(ckt->CKTstate0 + here->BJT2gpi);
            gmu= *(ckt->CKTstate0 + here->BJT2gmu);
            gm= *(ckt->CKTstate0 + here->BJT2gm);
            go= *(ckt->CKTstate0 + here->BJT2go);
            xgm=0;
            gx= *(ckt->CKTstate0 + here->BJT2gx);
            xcpi= *(ckt->CKTstate0 + here->BJT2cqbe);
            xcmu= *(ckt->CKTstate0 + here->BJT2cqbc);
            xcbx= *(ckt->CKTstate0 + here->BJT2cqbx);
            xcsub= *(ckt->CKTstate0 + here->BJT2cqsub); 
            xcmcb= *(ckt->CKTstate0 + here->BJT2cexbc);
            *(here->BJT2colColPtr) +=   (gcpr);
            *(here->BJT2baseBasePtr) +=   (gx) + (xcbx) * (s->real);
            *(here->BJT2baseBasePtr + 1) +=   (xcbx) * (s->imag);
            *(here->BJT2emitEmitPtr) +=   (gepr);
            
	    *(here->BJT2colPrimeColPrimePtr) +=   (gmu+go+gcpr)
                                        + (xcmu+xcbx) * (s->real);
            *(here->BJT2colPrimeColPrimePtr + 1) +=   (xcmu+xcbx)
                                                 * (s->imag);
						 
	    *(here->BJT2substConSubstPtr)      +=   (-xcsub) * (s->real);
	    *(here->BJT2substConSubstPtr = 1)  +=   (-xcsub) * (s->imag);
	    				 				 
						 
            *(here->BJT2basePrimeBasePrimePtr) +=   (gx+gpi+gmu)
                                   + (xcpi+xcmu+xcmcb) * (s->real);
            *(here->BJT2basePrimeBasePrimePtr + 1) +=   (xcpi+xcmu+xcmcb) 
                                                   * (s->imag);
            *(here->BJT2emitPrimeEmitPrimePtr) +=   (gpi+gepr+gm+go)
                                           + (xcpi+xgm) * (s->real);
            *(here->BJT2emitPrimeEmitPrimePtr + 1) +=   (xcpi+xgm)
                                                 * (s->imag);
            *(here->BJT2colColPrimePtr) +=   (-gcpr);
            *(here->BJT2baseBasePrimePtr) +=   (-gx);
            *(here->BJT2emitEmitPrimePtr) +=   (-gepr);
            *(here->BJT2colPrimeColPtr) +=   (-gcpr);
            *(here->BJT2colPrimeBasePrimePtr) +=   (-gmu+gm)
                                           +   (-xcmu+xgm) * (s->real);
            *(here->BJT2colPrimeBasePrimePtr + 1) +=   (-xcmu+xgm)
                                                * (s->imag);
            *(here->BJT2colPrimeEmitPrimePtr) +=   (-gm-go)
                                             + (-xgm) * (s->real);
            *(here->BJT2colPrimeEmitPrimePtr + 1) +=   (-xgm) *
                                                  (s->imag);
            *(here->BJT2basePrimeBasePtr) +=   (-gx);
            *(here->BJT2basePrimeColPrimePtr) +=   (-gmu)
                                        +   (-xcmu-xcmcb) * (s->real);
            *(here->BJT2basePrimeColPrimePtr + 1) +=   (-xcmu-xcmcb)
                                                * (s->imag);
            *(here->BJT2basePrimeEmitPrimePtr) +=   (-gpi)
                                              +  (-xcpi) * (s->real);
            *(here->BJT2basePrimeEmitPrimePtr + 1) +=   (-xcpi) 
                                                 * (s->imag);
            *(here->BJT2emitPrimeEmitPtr) +=   (-gepr);
            *(here->BJT2emitPrimeColPrimePtr) +=   (-go)
                                              +   (xcmcb) * (s->real);
            *(here->BJT2emitPrimeColPrimePtr + 1) +=   (xcmcb) 
                                               *  (s->imag);
            *(here->BJT2emitPrimeBasePrimePtr) +=   (-gpi-gm)
                                      +  (-xcpi-xgm-xcmcb) * (s->real);
            *(here->BJT2emitPrimeBasePrimePtr + 1) +=   (-xcpi-xgm-xcmcb)
                             * (s->imag);

/* 
 * Paolo Nenzi 2002 
 * 
 * This is a very quick and dirty hack to make BJ2 compile.
 * The code replacing the commented out one has been built using
 * the one in bjt2acld.c
 * DO NOT USE THIS DEVICE 
 * 
 *           *(here->BJT2substSubstPtr) +=   (xccs) * (s->real);
 *           *(here->BJT2substSubstPtr + 1) +=   (xccs) * (s->imag);
 *           *(here->BJT2colPrimeSubstPtr) +=   (-xccs) * (s->real);
 *           *(here->BJT2colPrimeSubstPtr + 1) +=   (-xccs) * (s->imag);
 *           *(here->BJT2substColPrimePtr) +=   (-xccs) * (s->real);
 *           *(here->BJT2substColPrimePtr + 1) +=   (-xccs) * (s->imag);
 */

            *(here->BJT2substSubstPtr)         +=   (xcsub)  * (s->real);
            *(here->BJT2substSubstPtr + 1)     +=   (xcsub)  * (s->imag);
	    *(here->BJT2substConSubstPtr)      +=   (-xcsub) * (s->real);
	    *(here->BJT2substConSubstPtr = 1)  +=   (-xcsub) * (s->imag);
	    *(here->BJT2substSubstConPtr)      +=   (-xcsub) * (s->real);
	    *(here->BJT2substSubstConPtr = 1)  +=   (-xcsub) * (s->imag);
	    
  
            *(here->BJT2baseColPrimePtr) +=   (-xcbx) * (s->real);
            *(here->BJT2baseColPrimePtr + 1) +=   (-xcbx) * (s->imag);
            *(here->BJT2colPrimeBasePtr) +=   (-xcbx) * (s->real);
            *(here->BJT2colPrimeBasePtr + 1) +=   (-xcbx) * (s->imag);
        }
    }
    return(OK);
}
