/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine deletes all BJTs from the circuit and frees
 * all storage they were using.
 */

#include "ngspice.h"
#include "bjtdefs.h"
#include "suffix.h"


void
BJTdestroy(GENmodel **inModel)

{

    BJTmodel **model = (BJTmodel**)inModel;
    BJTinstance *here;
    BJTinstance *prev = NULL;
    BJTmodel *mod = *model;
    BJTmodel *oldmod = NULL;

    for( ; mod ; mod = mod->BJTnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (BJTinstance *)NULL;
        for(here = mod->BJTinstances ; here ; here = here->BJTnextInstance) {
            if(prev){
                if(prev->BJTsens) FREE(prev->BJTsens);
                FREE(prev);
            }
            prev = here;
        }
        if(prev){
            if(prev->BJTsens) FREE(prev->BJTsens);
            FREE(prev);
        }
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
