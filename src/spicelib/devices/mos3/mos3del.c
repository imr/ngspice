/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos3defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS3delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance **fast = (MOS3instance **)inst;
    MOS3instance **prev = NULL;
    MOS3instance *here;

    for( ; model ; model = model->MOS3nextModel) {
        prev = &(model->MOS3instances);
        for(here = *prev; here ; here = *prev) {
            if(here->MOS3name == name || (fast && here==*fast) ) {
                *prev= here->MOS3nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MOS3nextInstance);
        }
    }
    return(E_NODEV);
}
