/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine deletes all VBICs from the circuit and frees
 * all storage they were using.
 */

#include "ngspice.h"
#include "vbicdefs.h"
#include "suffix.h"


void
VBICdestroy(GENmodel **inModel)
{

    VBICmodel **model = (VBICmodel**)inModel;
    VBICinstance *here;
    VBICinstance *prev = NULL;
    VBICmodel *mod = *model;
    VBICmodel *oldmod = NULL;

    for( ; mod ; mod = mod->VBICnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (VBICinstance *)NULL;
        for(here = mod->VBICinstances ; here ; here = here->VBICnextInstance) {
            if(prev){
                if(prev->VBICsens) FREE(prev->VBICsens);
                FREE(prev);
            }
            prev = here;
        }
        if(prev){
            if(prev->VBICsens) FREE(prev->VBICsens);
            FREE(prev);
        }
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
