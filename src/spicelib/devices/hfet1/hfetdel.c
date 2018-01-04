/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

/*
  Imported into hfeta model: Paolo Nenzi 2001
*/

#include "ngspice/ngspice.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HFETAdelete(GENmodel *model, IFuid name, GENinstance **kill)
{
    for (; model; model = model->GENnextModel) {
        GENinstance **prev = &(model->GENinstances);
        GENinstance *here = *prev;
        for (; here; here = *prev) {
            if (here->GENname == name || (kill && here == *kill)) {
                *prev = here->GENnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->GENnextInstance);
        }
    }

    return E_NODEV;
}
