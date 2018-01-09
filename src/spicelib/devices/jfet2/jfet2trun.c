/**********
Based on jfettrunc.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
JFET2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance *here;

    for( ; model != NULL; model = JFET2nextModel(model)) {
        for(here=JFET2instances(model);here!=NULL;here = JFET2nextInstance(here)){

            CKTterr(here->JFET2qgs,ckt,timeStep);
            CKTterr(here->JFET2qgd,ckt,timeStep);
        }
    }
    return(OK);
}
