/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

/*
 * This routine deletes a BJT2 model from the circuit and frees
 * the storage it was using.
 * returns an error if the model has instances
 */

#include "ngspice.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


int
BJT2mDelete(GENmodel **inModels, IFuid modname, GENmodel *kill)

{
    BJT2model **model = (BJT2model**)inModels;
    BJT2model *modfast = (BJT2model*)kill;

    BJT2model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->BJT2nextModel)) {
        if( (*model)->BJT2modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    if( (*model)->BJT2instances ) return(E_NOTEMPTY);
    *oldmod = (*model)->BJT2nextModel; /* cut deleted device out of list */
    FREE(*model);
    return(OK);

}
