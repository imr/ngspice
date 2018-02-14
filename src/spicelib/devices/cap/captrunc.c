/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

    for( ; model!= NULL; model = CAPnextModel(model)) {
        for(here = CAPinstances(model); here != NULL ;
                here = CAPnextInstance(here)) {

            CKTterr(here->CAPqcap,ckt,timeStep);
        }
    }
    return(OK);
}
