/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

/*
 * This routine performs truncation error calculations for
 * BJT2s in the circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


int
BJT2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)

{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;

    for( ; model != NULL; model = model->BJT2nextModel) {
        for(here=model->BJT2instances;here!=NULL;here = here->BJT2nextInstance){
            if (here->BJT2owner != ARCHme) continue;

            CKTterr(here->BJT2qbe,ckt,timeStep);
            CKTterr(here->BJT2qbc,ckt,timeStep);
            CKTterr(here->BJT2qsub,ckt,timeStep);
        }
    }
    return(OK);
}
