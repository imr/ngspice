/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/
/*
 */

#include "ngspice.h"
#include "resdefs.h"
#include "sperror.h"


int
RESdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance **fast = (RESinstance **)inst;
    RESinstance **prev = NULL;
    RESinstance *here;

    for( ; model ; model = model->RESnextModel) {
        prev = &(model->RESinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->RESname == name || (fast && here==*fast) ) {
                *prev= here->RESnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->RESnextInstance);
        }
    }
    return(E_NODEV);
}
