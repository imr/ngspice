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


/* ARGSUSED */
int
INDpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    INDmodel *model = (INDmodel*)inModel;
    double val;
    INDinstance *here;

    for( ; model != NULL; model = model->INDnextModel) {
        for( here = model->INDinstances;here != NULL; 
                here = here->INDnextInstance) {
	    
	    if (here->INDowner != ARCHme) continue;
    
            val = here->INDinduct;
	    
            *(here->INDposIbrptr) +=  1;
            *(here->INDnegIbrptr) -=  1;
            *(here->INDibrPosptr) +=  1;
            *(here->INDibrNegptr) -=  1;
            *(here->INDibrIbrptr ) -=   val * s->real;
            *(here->INDibrIbrptr +1) -= val * s->imag;
        }
    }
    return(OK);

}
