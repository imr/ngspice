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
#include "diodefs.h"
#include "suffix.h"


int
DIOpzLoad(inModel,ckt,s)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    SPcomplex *s;
{
    register DIOmodel *model = (DIOmodel*)inModel;
    double gspr;
    double geq;
    double xceq;
    register DIOinstance *here;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
	    if (here->DIOowner != ARCHme) continue;

            gspr=model->DIOresist*here->DIOarea;
            geq= *(ckt->CKTstate0 + here->DIOconduct);
            xceq= *(ckt->CKTstate0 + here->DIOcapCurrent);
            *(here->DIOposPosPtr ) += gspr;
            *(here->DIOnegNegPtr ) += geq + xceq * s->real;
            *(here->DIOnegNegPtr +1 ) += xceq * s->imag;
            *(here->DIOposPrimePosPrimePtr ) += geq + gspr + xceq * s->real;
            *(here->DIOposPrimePosPrimePtr +1 ) += xceq * s->imag;
            *(here->DIOposPosPrimePtr ) -= gspr;
            *(here->DIOnegPosPrimePtr ) -= geq + xceq * s->real;
            *(here->DIOnegPosPrimePtr +1 ) -= xceq * s->imag;
            *(here->DIOposPrimePosPtr ) -= gspr;
            *(here->DIOposPrimeNegPtr ) -= geq + xceq * s->real;
            *(here->DIOposPrimeNegPtr +1 ) -= xceq * s->imag;
        }
    }
    return(OK);

}
