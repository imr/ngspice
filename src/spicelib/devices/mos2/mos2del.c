/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos2defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS2delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance **fast = (MOS2instance **)inst;
    MOS2instance **prev = NULL;
    MOS2instance *here;

    for( ; model ; model = model->MOS2nextModel) {
        prev = &(model->MOS2instances);
        for(here = *prev; here ; here = *prev) {
            if(here->MOS2name == name || (fast && here==*fast) ) {
                *prev= here->MOS2nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MOS2nextInstance);
        }
    }
    return(E_NODEV);
}
