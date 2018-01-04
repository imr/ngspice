/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
TXLdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    TXLmodel *model = (TXLmodel *) inModel;
    TXLinstance **fast = (TXLinstance **) kill;
    TXLinstance **prev = NULL;
    TXLinstance *here;

    for (; model; model = model->TXLnextModel) {
        prev = &(model->TXLinstances);
        for (here = *prev; here; here = *prev) {
            if (here->TXLname == name || (fast && here == *fast)) {
                *prev = here->TXLnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->TXLnextInstance);
        }
    }

    return E_NODEV;
}
