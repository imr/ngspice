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

    for( ; model != NULL; model = INDnextModel(model)) {
        for( here = INDinstances(model);here != NULL; 
                here = INDnextInstance(here)) {
    
            val = here->INDinduct / here->INDm;
	    
            *(here->INDposIbrPtr) +=  1;
            *(here->INDnegIbrPtr) -=  1;
            *(here->INDibrPosPtr) +=  1;
            *(here->INDibrNegPtr) -=  1;
            *(here->INDibrIbrPtr ) -=   val * s->real;
            *(here->INDibrIbrPtr +1) -= val * s->imag;
        }
    }
    return(OK);

}
