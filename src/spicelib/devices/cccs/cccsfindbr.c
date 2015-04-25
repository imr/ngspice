/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCCSfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    CCCSmodel *model = (CCCSmodel *) inModel;
    CCCSinstance *here;

    for (; model; model = model->CCCSnextModel)
        for (here = model->CCCSinstances; here; here = here->CCCSnextInstance)
            if (here->CCCSname == name) {

                if (!here->CCCSbranch) {
                    CKTnode *tmp;
                    int error;

                    error = CKTmkCur(ckt, &tmp, here->CCCSname, "branch");
                    if (error)
                        return error;
                    here->CCCSbranch = tmp->number;
                }

                return here->CCCSbranch;
            }

    return 0;
}
