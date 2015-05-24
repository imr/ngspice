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
    double factor, rhs;

    NG_IGNORE(ckt);

    for (; model; model = model->ASRCnextModel) {
        for (here = model->ASRCinstances; here; here = here->ASRCnextInstance) {

            difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15; /* 300.15 !!!! */
            factor = 1.0
                + here->ASRCtc1 * difference
                + here->ASRCtc2 * difference * difference;

            if (here->ASRCreciproctc == 1)
                factor = 1 / factor;

            /*
             * Get the function and its derivatives from the
             * field in the instance structure. The field is
             * an array of doubles holding the rhs, and the
             * entries of the jacobian.
             */

            j = 0;

            if (here->ASRCtree->IFeval(here->ASRCtree, ckt->CKTgmin, &rhs, here->ASRCopValues, asrc_derivs) != OK)
                return(E_BADPARM);

            derivs = asrc_derivs;

            // mit factor multiplien !!

            if (here->ASRCtype == ASRC_VOLTAGE) {

                *(here->ASRCposptr[j++]) += 1.0;
                *(here->ASRCposptr[j++]) -= 1.0;
                *(here->ASRCposptr[j++]) -= 1.0;
                *(here->ASRCposptr[j++]) += 1.0;

                for (i = 0; i < here->ASRCtree->numVars; i++)
                    *(here->ASRCposptr[j++]) -= derivs[i] * factor;

            } else {

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    *(here->ASRCposptr[j++]) += derivs[i] * factor;
                    *(here->ASRCposptr[j++]) -= derivs[i] * factor;
                }

            }
        }
    }

    return(OK);
}
