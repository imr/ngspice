/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

/*
 * This routine deletes a BJT2 instance from the circuit and frees
 * the storage it was using.
 */

#include "ngspice.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


int
BJT2delete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance **fast = (BJT2instance**)kill;

    BJT2instance **prev = NULL;
    BJT2instance *here;

    for( ; model ; model = model->BJT2nextModel) {
        prev = &(model->BJT2instances);
        for(here = *prev; here ; here = *prev) {
            if(here->BJT2name == name || (fast && here==*fast) ) {
                *prev= here->BJT2nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->BJT2nextInstance);
        }
    }
    return(E_NODEV);
}
