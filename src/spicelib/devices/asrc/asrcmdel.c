/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

#include "ngspice.h"
#include <stdio.h>
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"


int
ASRCmDelete(modList,modname,killModel)
    GENmodel **modList;
    IFuid modname;
    GENmodel *killModel;

{

    register ASRCmodel **model = (ASRCmodel**)modList;
    register ASRCmodel *modfast = (ASRCmodel*)killModel;
    register ASRCinstance *here;
    register ASRCinstance *prev = NULL;
    register ASRCmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->ASRCnextModel)) {
        if( (*model)->ASRCmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->ASRCnextModel; /* cut deleted device out of list */
    for(here = (*model)->ASRCinstances ; here ; here = here->ASRCnextInstance) {
	FREE(here->ASRCacValues);
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
