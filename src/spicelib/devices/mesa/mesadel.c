/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
Imported into mesa model: 2001 Paolo Nenzi
 */

#include "ngspice/ngspice.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESAdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    MESAmodel *model = (MESAmodel*)inModel;
    MESAinstance **fast = (MESAinstance**)inst;
    MESAinstance **prev = NULL;
    MESAinstance *here;

    for( ; model ; model = model->MESAnextModel) {
        prev = &(model->MESAinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->MESAname == name || (fast && here==*fast) ) {
                *prev= here->MESAnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MESAnextInstance);
        }
    }
    return(E_NODEV);
}
