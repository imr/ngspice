/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddmdel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice.h"
#include "b3soidddef.h"
#include "sperror.h"
#include "suffix.h"

int
B3SOIDDmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
B3SOIDDmodel **model = (B3SOIDDmodel**)inModel;
B3SOIDDmodel *modfast = (B3SOIDDmodel*)kill;
B3SOIDDinstance *here;
B3SOIDDinstance *prev = NULL;
B3SOIDDmodel **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->B3SOIDDnextModel)) 
    {    if ((*model)->B3SOIDDmodName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->B3SOIDDnextModel; /* cut deleted device out of list */
    for (here = (*model)->B3SOIDDinstances; here; here = here->B3SOIDDnextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



