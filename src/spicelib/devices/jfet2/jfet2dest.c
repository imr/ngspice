/**********
Based on jfetdest.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/
/*
 */

#include "ngspice.h"
#include "jfet2defs.h"
#include "suffix.h"


void
JFET2destroy(GENmodel **inModel)
{
    JFET2model **model = (JFET2model**)inModel;
    JFET2instance *here;
    JFET2instance *prev = NULL;
    JFET2model *mod = *model;
    JFET2model *oldmod = NULL;

    for( ; mod ; mod = mod->JFET2nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (JFET2instance *)NULL;
        for(here = mod->JFET2instances ; here ; here = here->JFET2nextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
