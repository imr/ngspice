/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


#ifdef MUTUAL
int
MUTmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MUTmodel **model = (MUTmodel**)inModel;
    MUTmodel *modfast = (MUTmodel*)kill;
    MUTinstance *here;
    MUTinstance *prev = NULL;
    MUTmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->MUTnextModel)) {
        if( (*model)->MUTmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->MUTnextModel; /* cut deleted device out of list */
    for(here = (*model)->MUTinstances ; here ; here = here->MUTnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
#endif /* MUTUAL */
