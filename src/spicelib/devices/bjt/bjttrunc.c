/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine performs truncation error calculations for
 * BJTs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BJTtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)

{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;

    for( ; model != NULL; model = model->BJTnextModel) {
        for(here=model->BJTinstances;here!=NULL;here = here->BJTnextInstance){

            CKTterr(here->BJTqbe,ckt,timeStep);
            CKTterr(here->BJTqbc,ckt,timeStep);
            CKTterr(here->BJTqsub,ckt,timeStep);
        }
    }
    return(OK);
}
