/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice.h"
#include "mesdefs.h"
#include "sperror.h"
#include "suffix.h"


int
MESdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MESmodel *model = (MESmodel*)inModel;
    MESinstance **fast = (MESinstance**)inst;
    MESinstance **prev = NULL;
    MESinstance *here;

    for( ; model ; model = model->MESnextModel) {
        prev = &(model->MESinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->MESname == name || (fast && here==*fast) ) {
                *prev= here->MESnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MESnextInstance);
        }
    }
    return(E_NODEV);
}
