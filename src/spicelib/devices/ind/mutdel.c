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


#ifdef MUTUAL
int
MUTdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance **fast = (MUTinstance**)kill;
    MUTinstance **prev = NULL;
    MUTinstance *here;

    for( ; model ; model = model->MUTnextModel) {
        prev = &(model->MUTinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->MUTname == name || (fast && here==*fast) ) {
                *prev= here->MUTnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->MUTnextInstance);
        }
    }
    return(E_NODEV);
}
#endif /*MUTUAL*/
