/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "cpldefs.h"
#include "ngspice/suffix.h"

void
CPLdestroy(GENmodel **inModel)
{
    CPLmodel **model = (CPLmodel **)inModel;
    CPLinstance *here;
    CPLinstance *prev = NULL;
    CPLmodel *mod = *model;
    CPLmodel *oldmod = NULL;

    for( ; mod ; mod = mod->CPLnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = NULL;
        for(here = mod->CPLinstances ; here ; here = here->CPLnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
