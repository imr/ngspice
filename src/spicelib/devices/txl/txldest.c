/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice.h"
#include "txldefs.h"
#include "suffix.h"


void
TXLdestroy(GENmodel **inModel)
{
    TXLmodel **model = (TXLmodel **)inModel;
    TXLinstance *here;
    TXLinstance *prev = NULL;
    TXLmodel *mod = *model;
    TXLmodel *oldmod = NULL;

    for( ; mod ; mod = mod->TXLnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (TXLinstance *)NULL;
        for(here = mod->TXLinstances ; here ; here = here->TXLnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
