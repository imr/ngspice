/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice.h"
#include "capdefs.h"
#include "suffix.h"


void
CAPdestroy(GENmodel **inModel)

{

    CAPmodel **model = (CAPmodel**)inModel;
    CAPinstance *here;
    CAPinstance *prev = NULL;
    CAPmodel *mod = *model;
    CAPmodel *oldmod = NULL;

    for( ; mod ; mod = mod->CAPnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (CAPinstance *)NULL;
        for(here = mod->CAPinstances ; here ; here = here->CAPnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
