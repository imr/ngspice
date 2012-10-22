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

    for( ; model != NULL; model = model->CAPnextModel) {
        for( here = model->CAPinstances;here != NULL; 
                here = here->CAPnextInstance) {
	    
	    m = here -> CAPm;
    
            val = ckt->CKTomega * here->CAPcapac;
	    
            *(here->CAPposPosptr +1) += m * val;
            *(here->CAPnegNegptr +1) += m * val;
            *(here->CAPposNegptr +1) -= m * val;
            *(here->CAPnegPosptr +1) -= m * val;
        }
    }
    return(OK);

}

