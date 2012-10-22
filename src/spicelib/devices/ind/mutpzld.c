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


#ifdef MUTUAL
/* ARGSUSED */
int
MUTpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    MUTmodel *model = (MUTmodel*)inModel;
    double val;
    MUTinstance *here;

    NG_IGNORE(ckt);

    for( ; model != NULL; model = model->MUTnextModel) {
        for( here = model->MUTinstances;here != NULL; 
                here = here->MUTnextInstance) {
    
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
