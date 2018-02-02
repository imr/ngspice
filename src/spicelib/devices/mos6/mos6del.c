/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS6delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance **fast = (MOS6instance **)inst;
    MOS6instance **prev = NULL;
    MOS6instance *here;

    for( ; model ; model = model->MOS6nextModel) {
        prev = &(model->MOS6instances);
        for(here = *prev; here ; here = *prev) {
            if(here->MOS6name == name || (fast && here==*fast) ) {
                *prev= here->MOS6nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MOS6nextInstance);
        }
    }
    return(E_NODEV);
}
