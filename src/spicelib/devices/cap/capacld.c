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
CAPacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel*)inModel;
    double val;
    double m;
    CAPinstance *here;

    for( ; model != NULL; model = CAPnextModel(model)) {
        for( here = CAPinstances(model); here != NULL;
                here = CAPnextInstance(here)) {

            m = here->CAPm;

            val = ckt->CKTomega * here->CAPcapac;

            *(here->CAPposPosPtr +1) += m * val;
            *(here->CAPnegNegPtr +1) += m * val;
            *(here->CAPposNegPtr +1) -= m * val;
            *(here->CAPnegPosPtr +1) -= m * val;
        }
    }
    return(OK);

}

