/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ISRCacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    ISRCmodel *model = (ISRCmodel *) inModel;
    ISRCinstance *here;
    double m;

    for( ; model != NULL; model = ISRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = ISRCinstances(model); here != NULL ;
                here=ISRCnextInstance(here)) {

            double acReal, acImag;

            if (ckt->CKTmode & MODEACNOISE) {
                if ((GENinstance *) here == ckt->noise_input) {
                    acReal = 1.0;
                    acImag = 0.0;
                } else {
                    acReal = 0.0;
                    acImag = 0.0;
                }
            } else {
                acReal = here->ISRCacReal;
                acImag = here->ISRCacImag;
            }

            m = here->ISRCmValue;

            *(ckt->CKTrhs + (here->ISRCposNode)) +=
                m * acReal;
            *(ckt->CKTrhs + (here->ISRCnegNode)) -=
                m * acReal;
            *(ckt->CKTirhs + (here->ISRCposNode)) +=
                m * acImag;
            *(ckt->CKTirhs + (here->ISRCnegNode)) -=
                m * acImag;
        }
    }

    return(OK);
}
