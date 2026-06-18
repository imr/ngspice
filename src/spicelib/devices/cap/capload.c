/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

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
        for( ; model != NULL; model = CAPnextModel(model)) {

            /* loop through all the instances of the model */
            for (here = CAPinstances(model); here != NULL ;
                    here=CAPnextInstance(here)) {

                m = here->CAPm;

                if(cond1) {
                    vcap = here->CAPinitCond;
                } else {
                    vcap = *(ckt->CKTrhsOld+here->CAPposNode) -
                           *(ckt->CKTrhsOld+here->CAPnegNode) ;
                }
                if(ckt->CKTmode & (MODETRAN | MODEAC)) {
                    if (here->CAPdangling) {
                        /* Topology reduction: this cap hangs on a
                         * floating (degree-1) node.  Remove it from the system:
                         * pin the floating node(s) with a unit diagonal so the
                         * matrix stays nonsingular, and contribute no charge or
                         * current.  This eliminates the spurious LTE pressure
                         * that otherwise drives "Timestep too small" at the
                         * dangling node (set in CKTtopologyReduce()). */
                        if (here->CAPdangling & 1) *(here->CAPposPosPtr) += 1.0;
                        if (here->CAPdangling & 2) *(here->CAPnegNegPtr) += 1.0;
                        *(ckt->CKTstate0+here->CAPqcap) = 0.0;
                        continue;
                    }
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
                    /* Tiny conductance in parallel with the capacitor so that a
                     * node connected only through capacitors (e.g. a dangling
                     * compensation cap) still has a DC reference and cannot drift
                     * into a singular/ill-conditioned operating point.  This
                     * restores the 1e15 ohm "avoid floating nodes" resistor that
                     * the old behavioral C=f(v) expansion placed across the cap;
                     * 1e-15 S is negligible for caps on driven nodes. */
                    {
                        double gpar = 1e-15;
                        *(here->CAPposPosPtr) += m * geq + gpar;
                        *(here->CAPnegNegPtr) += m * geq + gpar;
                        *(here->CAPposNegPtr) -= m * geq + gpar;
                        *(here->CAPnegPosPtr) -= m * geq + gpar;
                    }
                    *(ckt->CKTrhs+here->CAPposNode) -= m * ceq;
                    *(ckt->CKTrhs+here->CAPnegNode) += m * ceq;
                } else
                    *(ckt->CKTstate0+here->CAPqcap) = here->CAPcapac * vcap;
            }
        }
    }
    return(OK);
}

