/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ISRCfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    ISRCmodel *model = (ISRCmodel *) inModel;
    ISRCinstance *here;

    for (; model; model = model->ISRCnextModel)
        for (here = model->ISRCinstances; here; here = here->ISRCnextInstance)
            if (here->ISRCname == name) {

                if (here->ISRCbranch == 0) {
                    CKTnode *tmp;
                    int error;

                    error = CKTmkCur(ckt, &tmp, here->ISRCname, "branch");
                    if (error)
                        return error;
                    here->ISRCbranch = tmp->number;
                }

                return here->ISRCbranch;
            }

    return 0;
}
