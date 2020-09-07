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

    for( ; model != NULL; model = BJTnextModel(model)) {
        for(here=BJTinstances(model);here!=NULL;here = BJTnextInstance(here)){

            CKTterr(here->BJTqbe,ckt,timeStep);
            CKTterr(here->BJTqbc,ckt,timeStep);
            CKTterr(here->BJTqsub,ckt,timeStep);
            if (model->BJTintCollResistGiven) {
                CKTterr(here->BJTqbcx,ckt,timeStep);
            }
        }
    }
    return(OK);
}
