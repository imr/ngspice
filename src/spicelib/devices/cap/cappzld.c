/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "capdefs.h"
#include "suffix.h"


/* ARGSUSED */
int
CAPpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)

{
    CAPmodel *model = (CAPmodel*)inModel;
    double val;
    double m;
    CAPinstance *here;

    for( ; model != NULL; model = model->CAPnextModel) {
        for( here = model->CAPinstances;here != NULL; 
                here = here->CAPnextInstance) {
	    
	    if (here->CAPowner != ARCHme) continue;
    
            val = here->CAPcapac;
            m = here->CAPm;
	    
	    *(here->CAPposPosptr ) +=   m * val * s->real;
            *(here->CAPposPosptr +1) += m * val * s->imag;
            *(here->CAPnegNegptr ) +=   m * val * s->real;
            *(here->CAPnegNegptr +1) += m * val * s->imag;
            *(here->CAPposNegptr ) -=   m * val * s->real;
            *(here->CAPposNegptr +1) -= m * val * s->imag;
            *(here->CAPnegPosptr ) -=   m * val * s->real;
            *(here->CAPnegPosptr +1) -= m * val * s->imag;
        }
    }
    return(OK);

}
