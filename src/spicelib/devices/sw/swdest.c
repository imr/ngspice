/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice.h"
#include "swdefs.h"
#include "suffix.h"


void
SWdestroy(GENmodel **inModel)
{
    SWmodel **model = (SWmodel**)inModel;
    SWinstance *here;
    SWinstance *prev = NULL;
    SWmodel *mod = *model;
    SWmodel *oldmod = NULL;

    for( ; mod ; mod = mod->SWnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (SWinstance *)NULL;
        for(here = mod->SWinstances ; here ; here = here->SWnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
