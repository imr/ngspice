/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "capdefs.h"
#include "suffix.h"


/* ARGSUSED */
int
CAPpzLoad(inModel,ckt,s)
    GENmodel *inModel;
    CKTcircuit *ckt;
    register SPcomplex *s;

{
    register CAPmodel *model = (CAPmodel*)inModel;
    double val;
    register CAPinstance *here;

    for( ; model != NULL; model = model->CAPnextModel) {
        for( here = model->CAPinstances;here != NULL; 
                here = here->CAPnextInstance) {
	    if (here->CAPowner != ARCHme) continue;
    
            val = here->CAPcapac;
            *(here->CAPposPosptr ) += val * s->real;
            *(here->CAPposPosptr +1) += val * s->imag;
            *(here->CAPnegNegptr ) += val * s->real;
            *(here->CAPnegNegptr +1) += val * s->imag;
            *(here->CAPposNegptr ) -= val * s->real;
            *(here->CAPposNegptr +1) -= val * s->imag;
            *(here->CAPnegPosptr ) -= val * s->real;
            *(here->CAPnegPosptr +1) -= val * s->imag;
        }
    }
    return(OK);

}
