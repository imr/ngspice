/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos1defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS1delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance **fast = (MOS1instance **)inst;
    MOS1instance **prev = NULL;
    MOS1instance *here;

    for( ; model ; model = model->MOS1nextModel) {
        prev = &(model->MOS1instances);
        for(here = *prev; here ; here = *prev) {
            if(here->MOS1name == name || (fast && here==*fast) ) {
                *prev= here->MOS1nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MOS1nextInstance);
        }
    }
    return(E_NODEV);
}
