/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice.h"
#include "swdefs.h"
#include "sperror.h"
#include "suffix.h"


int
SWdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    SWmodel *model = (SWmodel *)inModel;
    SWinstance **fast = (SWinstance **)inst;
    SWinstance **prev = NULL;
    SWinstance *here;

    for( ; model ; model = model->SWnextModel) {
        prev = &(model->SWinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->SWname == name || (fast && here==*fast) ) {
                *prev= here->SWnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->SWnextInstance);
        }
    }
    return(E_NODEV);
}
