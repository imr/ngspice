/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice.h"
#include "mos9defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS9delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance **fast = (MOS9instance **)inst;
    MOS9instance **prev = NULL;
    MOS9instance *here;

    for( ; model ; model = model->MOS9nextModel) {
        prev = &(model->MOS9instances);
        for(here = *prev; here ; here = *prev) {
            if(here->MOS9name == name || (fast && here==*fast) ) {
                *prev= here->MOS9nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MOS9nextInstance);
        }
    }
    return(E_NODEV);
}
