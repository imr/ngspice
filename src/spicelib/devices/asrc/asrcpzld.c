/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/complex.h"


/* actually load the current voltage value into the
 * sparse matrix previously provided
 */

int
ASRCpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    double value;
    int i, j;
    double difference;
    double factor;

    NG_IGNORE(s);

    for (; model; model = model->ASRCnextModel) {
        for (here = model->ASRCinstances; here; here = here->ASRCnextInstance) {

            difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15;
            factor = 1.0
                + here->ASRCtc1 * difference
                + here->ASRCtc2 * difference * difference;

            if (here->ASRCreciproctc == 1)
                factor = 1 / factor;

            j = 0;

            /* Get the function evaluated and the derivatives too */
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

            /* Fill the vector of values from the previous solution */
            for (i = 0; i < here->ASRCtree->numVars; i++)
                if (here->ASRCtree->varTypes[i] == IF_INSTANCE) {
                    int branch = CKTfndBranch(ckt, here->ASRCtree->vars[i].uValue);
                    asrc_vals[i] = *(ckt->CKTrhsOld + branch);
                } else {
                    int node_num = (here->ASRCtree->vars[i].nValue) -> number;
                    asrc_vals[i] = *(ckt->CKTrhsOld + node_num);
                }

            if (here->ASRCtree->IFeval(here->ASRCtree, ckt->CKTgmin, &value, asrc_vals, asrc_derivs) != OK)
                return(E_BADPARM);

            if (here->ASRCtype == ASRC_VOLTAGE) {

                *(here->ASRCposPtr[j++]) += 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) -= 1.0;
                *(here->ASRCposPtr[j++]) += 1.0;

                for (i = 0; i < here->ASRCtree->numVars; i++)
                    *(here->ASRCposPtr[j++]) -= asrc_derivs[i] / factor;

            } else {

                for (i = 0; i < here->ASRCtree->numVars; i++) {
                    *(here->ASRCposPtr[j++]) += asrc_derivs[i] / factor;
                    *(here->ASRCposPtr[j++]) -= asrc_derivs[i] / factor;
                }

            }
        }
    }

    return(OK);
}
