/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "urcdefs.h"
#include "sperror.h"
#include "suffix.h"


int
URCdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    URCmodel *model = (URCmodel *)inModel;
    URCinstance **fast = (URCinstance**)inst;
    URCinstance **prev = NULL;
    URCinstance *here;

    for( ; model ; model = model->URCnextModel) {
        prev = &(model->URCinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->URCname == name || (fast && here==*fast) ) {
                *prev= here->URCnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->URCnextInstance);
        }
    }
    return(E_NODEV);
}
