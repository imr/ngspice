/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "capdefs.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
CAPpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)

{
    CAPmodel *model = (CAPmodel*)inModel;
    double val;
    double m;
    CAPinstance *here;

    NG_IGNORE(ckt);

    for( ; model != NULL; model = CAPnextModel(model)) {
        for( here = CAPinstances(model);here != NULL; 
                here = CAPnextInstance(here)) {
    
            val = here->CAPcapac;
            m = here->CAPm;
	    
	    *(here->CAPposPosPtr ) +=   m * val * s->real;
            *(here->CAPposPosPtr +1) += m * val * s->imag;
            *(here->CAPnegNegPtr ) +=   m * val * s->real;
            *(here->CAPnegNegPtr +1) += m * val * s->imag;
            *(here->CAPposNegPtr ) -=   m * val * s->real;
            *(here->CAPposNegPtr +1) -= m * val * s->imag;
            *(here->CAPnegPosPtr ) -=   m * val * s->real;
            *(here->CAPnegPosPtr +1) -= m * val * s->imag;
        }
    }
    return(OK);

}
