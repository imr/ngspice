/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "inddefs.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
INDpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    INDmodel *model = (INDmodel*)inModel;
    double val;
    INDinstance *here;

    NG_IGNORE(ckt);

    for( ; model != NULL; model = model->INDnextModel) {
        for( here = model->INDinstances;here != NULL; 
                here = here->INDnextInstance) {
    
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
