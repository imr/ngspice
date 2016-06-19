/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


double *asrc_vals, *asrc_derivs;
int    asrc_nvals;


/* actually load the current voltage value into the
 * sparse matrix previously provided
 */

int
ASRCload(GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    int i, j;
    double rhs;
    double difference;
    double factor;

    for (; model; model = model->ASRCnextModel) {
        for (here = model->ASRCinstances; here; here=here->ASRCnextInstance) {

            difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15;
            factor = 1.0
                + here->ASRCtc1 * difference
                + here->ASRCtc2 * difference * difference;

            if (here->ASRCreciproctc == 1)
                factor = 1 / factor;

            /*
             * Get the function and its derivatives evaluated
             */
            i = here->ASRCtree->numVars;
            if (asrc_nvals < i) {
                if (asrc_nvals) {
                    FREE(asrc_vals);
                    FREE(asrc_derivs);
                }
                asrc_nvals = i;
                asrc_vals = TMALLOC(double, i);
                asrc_derivs = TMALLOC(double, i);
            }

            j = 0;

            /*
             * Fill the vector of values from the previous solution
             */
            for (i = 0; i < here->ASRCtree->numVars; i++)
                asrc_vals[i] = ckt->CKTrhsOld[here->ASRCvars[i]];

            if (here->ASRCtree->IFeval(here->ASRCtree, ckt->CKTgmin, &rhs, asrc_vals, asrc_derivs) != OK)
                return(E_BADPARM);

            /* The convergence test */
            here->ASRCprev_value = rhs;

            /* The ac load precomputation and storage */
            if (ckt->CKTmode & MODEINITSMSIG)
                for (i = 0; i < here->ASRCtree->numVars; i++)
                    here->ASRCacValues[i] = asrc_derivs[i];

            if (here->ASRCtype == ASRC_VOLTAGE) {

                *(here->ASRCposPtr[j++]) += 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) += 1.0;

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    rhs -= (asrc_vals[i] * asrc_derivs[i]);

                    *(here->ASRCposPtr[j++]) -= asrc_derivs[i] * factor;
                }

                ckt->CKTrhs[here->ASRCbranch] += factor * rhs;

            } else {

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    rhs -= (asrc_vals[i] * asrc_derivs[i]);

                    *(here->ASRCposPtr[j++]) += asrc_derivs[i] * factor;
                    *(here->ASRCposPtr[j++]) -= asrc_derivs[i] * factor;
                }

                ckt->CKTrhs[here->ASRCposNode] -= factor * rhs;
                ckt->CKTrhs[here->ASRCnegNode] += factor * rhs;
            }

            /* Store the rhs for small signal analysis */
            if (ckt->CKTmode & MODEINITSMSIG)
                here->ASRCacValues[here->ASRCtree->numVars] = factor * rhs;
        }
    }

    return(OK);
}
