/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int INDfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name) {
    INDmodel *model = (INDmodel *)inModel;
    INDinstance *here;
    int error;
    CKTnode *tmp;

    for (; model != NULL; model = model->INDnextModel) {
        for (here = model->INDinstances; here != NULL; here = here->INDnextInstance) {
            if (here->INDname == name) {
                if (here->INDbrEq == 0) {
                    error = CKTmkCur(ckt, &tmp, here->INDname, "branch");
                    if (error) return error;
                    here->INDbrEq = tmp->number;
                }
                return here->INDbrEq;
            }
        }
    }
    return 0;
}
