/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


int
INDtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    for( ; model!= NULL; model = model->INDnextModel) {
        for(here = model->INDinstances ; here != NULL ;
                here = here->INDnextInstance) {
	    if (here->INDowner != ARCHme) continue;

            CKTterr(here->INDflux,ckt,timeStep);
        }
    }
    return(OK);
}
