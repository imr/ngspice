/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine deletes a BJT model from the circuit and frees
 * the storage it was using.
 * returns an error if the model has instances
 */

#include "ngspice.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BJTmDelete(GENmodel **inModels, IFuid modname, GENmodel *kill)
{
    BJTmodel **model = (BJTmodel**)inModels;
    BJTmodel *modfast = (BJTmodel*)kill;

    BJTmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->BJTnextModel)) {
        if( (*model)->BJTmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    if( (*model)->BJTinstances ) return(E_NOTEMPTY);
    *oldmod = (*model)->BJTnextModel; /* cut deleted device out of list */
    FREE(*model);
    return(OK);

}
