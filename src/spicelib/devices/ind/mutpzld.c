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
MUTpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    MUTmodel *model = (MUTmodel*)inModel;
    double val;
    MUTinstance *here;

    NG_IGNORE(ckt);

    for( ; model != NULL; model = MUTnextModel(model)) {
        for( here = MUTinstances(model);here != NULL; 
                here = MUTnextInstance(here)) {
    
            val =  here->MUTfactor;
            *(here->MUTbr1br2Ptr ) -= val * s->real;
            *(here->MUTbr1br2Ptr +1) -= val * s->imag;
            *(here->MUTbr2br1Ptr ) -= val * s->real;
            *(here->MUTbr2br1Ptr +1) -= val * s->imag;
        }
    }
    return(OK);

}
