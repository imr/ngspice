/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ccvsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CCVSdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance **fast = (CCVSinstance**)kill;
    CCVSinstance **prev = NULL;
    CCVSinstance *here;

    for( ; model ; model = model->CCVSnextModel) {
        prev = &(model->CCVSinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->CCVSname == name || (fast && here==*fast) ) {
                *prev= here->CCVSnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->CCVSnextInstance);
        }
    }
    return(E_NODEV);
}
