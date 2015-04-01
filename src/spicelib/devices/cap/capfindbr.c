/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* macro to make elements with built-in test for out of memory */
#define TSTALLOC(matrix, ptr, row, col) \
do { if((ptr = SMPmakeElt(matrix, row, col)) == NULL){\
    return E_NOMEM;\
} } while(0)

int CAPfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name) {
    CAPmodel *model = (CAPmodel *)inModel;
    CAPinstance *here;
    int error;
    CKTnode *tmp;

    for (; model != NULL; model = model->CAPnextModel) {
        for (here = model->CAPinstances; here != NULL; here = here->CAPnextInstance) {
            if (here->CAPname == name) {
                if (here->CAPbrptr == NULL) {
                    error = CKTmkCur(ckt, &tmp, here->CAPname, "branch");
                    if (error) return error;
                    here->CAPbrEq = tmp->number; 
                    if (ckt->CKTmatrix != NULL)
                        TSTALLOC(ckt->CKTmatrix, here->CAPbrptr, here->CAPbrEq, here->CAPbrEq);
                }
                return here->CAPbrEq;
            }
        }
    }
    return 0;
}
