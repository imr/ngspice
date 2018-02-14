/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
INDtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    for( ; model!= NULL; model = INDnextModel(model)) {
        for(here = INDinstances(model); here != NULL ;
                here = INDnextInstance(here)) {

            CKTterr(here->INDflux,ckt,timeStep);
        }
    }
    return(OK);
}
