/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCfindBr(CKTcircuit *ckt, GENmodel *inputModel, IFuid name)
{
    ASRCmodel *model = (ASRCmodel*) inputModel;
    ASRCinstance *here;
    int error;
    CKTnode *tmp;

    for (; model; model = ASRCnextModel(model))
        for (here = ASRCinstances(model); here; here = ASRCnextInstance(here))
            if (here->ASRCname == name) {
                if (here->ASRCbranch == 0) {
                    error = CKTmkCur(ckt, &tmp, here->ASRCname, "branch");
                    if (error)
                        return(error);
                    here->ASRCbranch = tmp->number;
                }
                return(here->ASRCbranch);
            }

    return(0);
}
