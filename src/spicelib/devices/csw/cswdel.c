/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice.h"
#include "cswdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CSWdelete(GENmodel *inModel, IFuid name, GENinstance **inst)

{
    CSWmodel *model = (CSWmodel*)inModel;
    CSWinstance **fast = (CSWinstance**)inst;
    CSWinstance **prev = NULL;
    CSWinstance *here;

    for( ; model ; model = model->CSWnextModel) {
        prev = &(model->CSWinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->CSWname == name || (fast && here==*fast) ) {
                *prev= here->CSWnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->CSWnextInstance);
        }
    }
    return(E_NODEV);
}
