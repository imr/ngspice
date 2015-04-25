/***********************************
* Author: Marcel Hendrix, Feb 2015 *
***********************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCCSfindBr (CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    VCCSmodel *model = (VCCSmodel *) inModel;
    VCCSinstance *here;

    for (; model; model = model->VCCSnextModel)
        for (here = model->VCCSinstances; here; here = here->VCCSnextInstance)
            if (here->VCCSname == name) {

                if (!here->VCCSbranch) {
                    CKTnode *tmp;
                    int error;

                    error = CKTmkCur(ckt, &tmp, here->VCCSname, "branch");
                    if (error)
                        return error;
                    here->VCCSbranch = tmp->number;
                }

                return here->VCCSbranch;
            }

    return 0;
}
