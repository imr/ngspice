/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "isrcdefs.h"
#include "sperror.h"
#include "suffix.h"


int
ISRCdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    ISRCmodel *model = (ISRCmodel*)inModel;
    ISRCinstance **fast = (ISRCinstance**)inst;
    ISRCinstance **prev = NULL;
    ISRCinstance *here;

    for( ; model ; model = model->ISRCnextModel) {
        prev = &(model->ISRCinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->ISRCname == name || (fast && here==*fast) ) {
                *prev= here->ISRCnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->ISRCnextInstance);
        }
    }
    return(E_NODEV);
}
