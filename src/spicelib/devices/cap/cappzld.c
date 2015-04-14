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

    for( ; model != NULL; model = model->CAPnextModel) {
        for( here = model->CAPinstances;here != NULL; 
                here = here->CAPnextInstance) {
    
            val = here->CAPcapac;
            m = here->CAPm;
	    
            *(here->CAP_posPrime_posPrime )   += m * val * s->real;
            *(here->CAP_posPrime_posPrime +1) += m * val * s->imag;
            *(here->CAP_neg_neg )             += m * val * s->real;
            *(here->CAP_neg_neg +1)           += m * val * s->imag;
            *(here->CAP_posPrime_neg )        -= m * val * s->real;
            *(here->CAP_posPrime_neg +1)      -= m * val * s->imag;
            *(here->CAP_neg_PosPrime )        -= m * val * s->real;
            *(here->CAP_neg_PosPrime +1)      -= m * val * s->imag;

            if (here->CAPbranch) {
                *(here->CAP_pos_ibr)      += 1.0;
                *(here->CAP_posPrime_ibr) -= 1.0;
                *(here->CAP_ibr_pos)      += 1.0;
                *(here->CAP_ibr_posPrime) -= 1.0;
            }
        }
    }
    return(OK);

}
