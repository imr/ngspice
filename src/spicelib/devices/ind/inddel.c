/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


int
INDdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance **fast = (INDinstance**)kill;
    INDinstance **prev = NULL;
    INDinstance *here;

    for( ; model ; model = model->INDnextModel) {
        prev = &(model->INDinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->INDname == name || (fast && here==*fast) ) {
                *prev= here->INDnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->INDnextInstance);
        }
    }
    return(E_NODEV);
}
