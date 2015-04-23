/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    CAPmodel *model = (CAPmodel *) inModel;
    CAPinstance *here;

    for (; model; model = model->CAPnextModel)
        for (here = model->CAPinstances; here; here = here->CAPnextInstance)
            if (here->CAPname == name) {

                if (here->CAPbranch == 0) {
                    CKTnode *tmp;
                    int error;

                    error = CKTmkCur(ckt, &tmp, here->CAPname, "branch");
                    if (error)
                        return error;
                    here->CAPbranch = tmp->number;
                }

                return here->CAPbranch;
            }

    return 0;
}
