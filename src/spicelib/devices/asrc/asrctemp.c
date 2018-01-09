/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"


int
ASRCtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel *) inModel;
    ASRCinstance *here;

    for (; model; model = ASRCnextModel(model)) {
        for (here = ASRCinstances(model); here; here = ASRCnextInstance(here)) {

            /* Default Value Processing for Source Instance */

            if (!here->ASRCtempGiven) {
                here->ASRCtemp = ckt->CKTtemp;
                if (!here->ASRCdtempGiven)
                    here->ASRCdtemp = 0.0;
            } else {
                here->ASRCdtemp = 0.0;
                if (here->ASRCdtempGiven)
                    printf("%s: Instance temperature specified, dtemp ignored\n", here->ASRCname);
            }

        }
    }

    return(OK);
}
