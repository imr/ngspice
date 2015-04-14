/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
RESfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    RESmodel *model = (RESmodel *) inModel;
    RESinstance *here;

    for (; model; model = model->RESnextModel)
        for (here = model->RESinstances; here; here = here->RESnextInstance)
            if (here->RESname == name) {

                if (!here->RESbranch) {
                    CKTnode *tmp;
                    int error;

                    error = CKTmkCur(ckt, &tmp, here->RESname, "branch");
                    if (error)
                        return error;
                    here->RESbranch = tmp->number;

                    error = CKTmkVolt(ckt, &tmp, here->RESname, "aux");
                    if (error)
                        return error;
                    here->RESposPrimeNode = tmp->number;
                }

                return here->RESbranch;
            }

    return 0;
}
