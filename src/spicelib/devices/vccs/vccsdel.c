/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vccsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCCSdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance **fast = (VCCSinstance**)inst;
    VCCSinstance **prev = NULL;
    VCCSinstance *here;

    for( ; model ; model = model->VCCSnextModel) {
        prev = &(model->VCCSinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->VCCSname == name || (fast && here==*fast) ) {
                *prev= here->VCCSnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->VCCSnextInstance);
        }
    }
    return(E_NODEV);
}
