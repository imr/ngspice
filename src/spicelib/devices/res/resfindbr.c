/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* macro to make elements with built-in test for out of memory */
#define TSTALLOC(matrix, ptr, row, col)                         \
    do {                                                        \
        if((ptr = SMPmakeElt(matrix, row, col)) == NULL)        \
            return E_NOMEM;                                     \
    } while(0)


int
RESfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    RESmodel *model = (RESmodel *) inModel;
    RESinstance *here;
    int error;
    CKTnode *tmp;

    for (; model != NULL; model = model->RESnextModel)
        for (here = model->RESinstances; here != NULL; here = here->RESnextInstance) {
            if (here->RESname == name) {
                if (here->RESbrptr == NULL) {
                    error = CKTmkCur(ckt, &tmp, here->RESname, "branch");
                    if (error)
                        return error;
                    here->RESbrEq = tmp->number;
                    if (ckt->CKTmatrix != NULL)
                        TSTALLOC(ckt->CKTmatrix, here->RESbrptr, here->RESbrEq, here->RESbrEq);
                }
                return here->RESbrEq;
            }
        }

    return 0;
}
