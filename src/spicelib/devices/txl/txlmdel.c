/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice.h"
#include "txldefs.h"
#include "sperror.h"
#include "suffix.h"


int
TXLmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    TXLmodel **model = (TXLmodel **)inModel;
    TXLmodel *modfast = (TXLmodel *)kill;
    TXLinstance *here;
    TXLinstance *prev = NULL;
    TXLmodel **oldmod;
    oldmod = model;

    for( ; *model ; model = &((*model)->TXLnextModel)) {
        if( (*model)->TXLmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->TXLnextModel; /* cut deleted device out of list */
    for(here = (*model)->TXLinstances ; here ; here = here->TXLnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
