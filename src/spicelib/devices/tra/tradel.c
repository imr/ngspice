/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "tradefs.h"
#include "sperror.h"
#include "suffix.h"


int
TRAdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    TRAinstance **fast = (TRAinstance **)kill;
    TRAmodel *model = (TRAmodel *)inModel;
    TRAinstance **prev = NULL;
    TRAinstance *here;

    for( ; model ; model = model->TRAnextModel) {
        prev = &(model->TRAinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->TRAname == name || (fast && here==*fast) ) {
                *prev= here->TRAnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->TRAnextInstance);
        }
    }
    return(E_NODEV);
}
