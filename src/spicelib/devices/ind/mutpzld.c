/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "inddefs.h"
#include "suffix.h"


#ifdef MUTUAL
/* ARGSUSED */
int
MUTpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    MUTmodel *model = (MUTmodel*)inModel;
    double val;
    MUTinstance *here;

    for( ; model != NULL; model = model->MUTnextModel) {
        for( here = model->MUTinstances;here != NULL; 
                here = here->MUTnextInstance) {
	    if (here->MUTowner != ARCHme) continue;
    
            val =  here->MUTfactor;
            *(here->MUTbr1br2 ) -= val * s->real;
            *(here->MUTbr1br2 +1) -= val * s->imag;
            *(here->MUTbr2br1 ) -= val * s->real;
            *(here->MUTbr2br1 +1) -= val * s->imag;
        }
    }
    return(OK);

}
#endif /*MUTUAL*/
