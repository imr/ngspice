/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soidddel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
B3SOIDDdelete(GENmodel *model, IFuid name, GENinstance **kill)
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
