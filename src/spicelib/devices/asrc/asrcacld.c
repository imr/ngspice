/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*
 * Actually load the current voltage value into the
 * sparse matrix previously provided. The values have
 * been precomputed and stored with the instance model.
 */

int
ASRCacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    int i, j;
    double *derivs;
    double difference;
    double factor;

    NG_IGNORE(ckt);

    for (; model; model = ASRCnextModel(model)) {
        for (here = ASRCinstances(model); here; here = ASRCnextInstance(here)) {

            difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15;
            factor = 1.0
                + here->ASRCtc1 * difference
                + here->ASRCtc2 * difference * difference;

            if (here->ASRCreciproctc == 1)
                factor = 1 / factor;

            if (here->ASRCreciprocm == 1)
                factor = factor / here->ASRCm;
            else
                factor = factor * here->ASRCm;

            /*
             * Get the function and its derivatives from the
             * field in the instance structure. The field is
             * an array of doubles holding the rhs, and the
             * entries of the jacobian.
             */

            j = 0;
            derivs = here->ASRCacValues;

            if (here->ASRCtype == ASRC_VOLTAGE) {

                *(here->ASRCposPtr[j++]) += 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) += 1.0;

                for (i = 0; i < here->ASRCtree->numVars; i++)
                    *(here->ASRCposPtr[j++]) -= derivs[i] * factor;

            } else {

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    *(here->ASRCposPtr[j++]) += derivs[i] * factor;
                    *(here->ASRCposPtr[j++]) -= derivs[i] * factor;
                }

            }
        }
    }

    return(OK);
}
