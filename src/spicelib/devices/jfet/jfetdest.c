/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "jfetdefs.h"
#include "suffix.h"


void
JFETdestroy(GENmodel **inModel)
{
    JFETmodel **model = (JFETmodel**)inModel;
    JFETinstance *here;
    JFETinstance *prev = NULL;
    JFETmodel *mod = *model;
    JFETmodel *oldmod = NULL;

    for( ; mod ; mod = mod->JFETnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (JFETinstance *)NULL;
        for(here = mod->JFETinstances ; here ; here = here->JFETnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
