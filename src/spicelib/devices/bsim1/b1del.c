/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "bsim1def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
B1delete(GENmodel *inModel, IFuid name, GENinstance **inInst)

{

    B1instance **fast = (B1instance**)inInst;
    B1model *model = (B1model*)inModel;
    B1instance **prev = NULL;
    B1instance *here;

    for( ; model ; model = model->B1nextModel) {
        prev = &(model->B1instances);
        for(here = *prev; here ; here = *prev) {
            if(here->B1name == name || (fast && here==*fast) ) {
                *prev= here->B1nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->B1nextInstance);
        }
    }
    return(E_NODEV);
}
