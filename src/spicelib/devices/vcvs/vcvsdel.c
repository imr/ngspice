/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vcvsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCVSdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance **fast = (VCVSinstance**)inst;
    VCVSinstance **prev = NULL;
    VCVSinstance *here;

    for( ; model ; model = model->VCVSnextModel) {
        prev = &(model->VCVSinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->VCVSname == name || (fast && here==*fast) ) {
                *prev= here->VCVSnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->VCVSnextInstance);
        }
    }
    return(E_NODEV);
}
