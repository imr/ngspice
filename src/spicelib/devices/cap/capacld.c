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
        for( here = model->CAPinstances; here != NULL;
                here = here->CAPnextInstance) {

            m = here->CAPm;

            val = ckt->CKTomega * here->CAPcapac;

            *(here->CAP_posPrime_pos +1) += m * val;
            *(here->CAP_neg_neg +1)      += m * val;
            *(here->CAP_posPrime_neg +1) -= m * val;
            *(here->CAP_neg_pos +1)      -= m * val;

            if (here->CAPbranch) {
                *(here->CAP_pos_ibr)      += 1.0;
                *(here->CAP_posPrime_ibr) -= 1.0;
            }
        }
    }
    return(OK);

}

