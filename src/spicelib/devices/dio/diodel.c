/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


int
DIOdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance **fast = (DIOinstance**)kill;
    DIOinstance **prev = NULL;
    DIOinstance *here;

    for( ; model ; model = model->DIOnextModel) {
        prev = &(model->DIOinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->DIOname == name || (fast && here==*fast) ) {
                *prev= here->DIOnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->DIOnextInstance);
        }
    }
    return(E_NODEV);
}
