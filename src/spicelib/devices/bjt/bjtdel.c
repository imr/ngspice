/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * This routine deletes a BJT instance from the circuit and frees
 * the storage it was using.
 */

#include "ngspice.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BJTdelete(GENmodel *inModel, IFuid name, GENinstance **kill)

{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance **fast = (BJTinstance**)kill;

    BJTinstance **prev = NULL;
    BJTinstance *here;

    for( ; model ; model = model->BJTnextModel) {
        prev = &(model->BJTinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->BJTname == name || (fast && here==*fast) ) {
                *prev= here->BJTnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->BJTnextInstance);
        }
    }
    return(E_NODEV);
}
