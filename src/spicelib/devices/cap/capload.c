/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "capdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
CAPload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current capacitance value into the 
         * sparse matrix previously provided 
         */
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    int cond1;
    double vcap;
    double geq;
    double ceq;
    int error;
    double m;

    /* check if capacitors are in the circuit or are open circuited */
    if(ckt->CKTmode & (MODETRAN|MODEAC|MODETRANOP) ) {
        /* evaluate device independent analysis conditions */
        cond1= 
            ( ( (ckt->CKTmode & MODEDC) && 
                (ckt->CKTmode & MODEINITJCT) )
            || ( ( ckt->CKTmode & MODEUIC) &&
                ( ckt->CKTmode & MODEINITTRAN) ) ) ;
        /*  loop through all the capacitor models */
        for( ; model != NULL; model = model->CAPnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->CAPinstances; here != NULL ;
                    here=here->CAPnextInstance) {
		
		if (here->CAPowner != ARCHme) continue;
                
		m = here->CAPm;
		
                if(cond1) {
                    vcap = here->CAPinitCond;
                } else {
                    vcap = *(ckt->CKTrhsOld+here->CAPposNode) - 
                        *(ckt->CKTrhsOld+here->CAPnegNode) ;   
                }
                if(ckt->CKTmode & (MODETRAN | MODEAC)) {
#ifndef PREDICTOR
                    if(ckt->CKTmode & MODEINITPRED) {
                        *(ckt->CKTstate0+here->CAPqcap) = 
                            *(ckt->CKTstate1+here->CAPqcap);
                    } else { /* only const caps - no poly's */
#endif /* PREDICTOR */
                        *(ckt->CKTstate0+here->CAPqcap) = here->CAPcapac * vcap;
                        if((ckt->CKTmode & MODEINITTRAN)) {
                            *(ckt->CKTstate1+here->CAPqcap) = 
                                *(ckt->CKTstate0+here->CAPqcap);
                        }
#ifndef PREDICTOR
                    }
#endif /* PREDICTOR */
                    error = NIintegrate(ckt,&geq,&ceq,here->CAPcapac,
                            here->CAPqcap);
                    if(error) return(error);
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1+here->CAPccap) = 
                            *(ckt->CKTstate0+here->CAPccap);
                    }
                    *(here->CAPposPosptr) += m * geq;
                    *(here->CAPnegNegptr) += m * geq;
                    *(here->CAPposNegptr) -= m * geq;
                    *(here->CAPnegPosptr) -= m * geq;
                    *(ckt->CKTrhs+here->CAPposNode) -= m * ceq;
                    *(ckt->CKTrhs+here->CAPnegNode) += m * ceq;
                } else
		    *(ckt->CKTstate0+here->CAPqcap) = here->CAPcapac * vcap;
            }
        }
    }
    return(OK);
}

