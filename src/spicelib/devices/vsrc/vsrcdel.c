/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vsrcdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VSRCdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance **fast = (VSRCinstance**)inst;
    VSRCinstance **prev = NULL;
    VSRCinstance *here;

    for( ; model ; model = model->VSRCnextModel) {
        prev = &(model->VSRCinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->VSRCname == name || (fast && here==*fast) ) {
                *prev= here->VSRCnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->VSRCnextInstance);
        }
    }
    return(E_NODEV);
}
