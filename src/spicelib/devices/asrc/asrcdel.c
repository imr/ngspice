/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCdelete(GENmodel *model, IFuid name, GENinstance **fast)
{
    ASRCinstance **instPtr = (ASRCinstance**) fast;
    ASRCmodel *modPtr = (ASRCmodel*) model;

    ASRCinstance **prev = NULL;
    ASRCinstance *here;

    for (; modPtr ; modPtr = modPtr->ASRCnextModel) {
        prev = &(modPtr->ASRCinstances);
        for (here = *prev; here ; here = *prev) {
            if (here->ASRCname == name || (instPtr && here == *instPtr)) {
                *prev = here->ASRCnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->ASRCnextInstance);
        }
    }

    return(E_NODEV);
}
