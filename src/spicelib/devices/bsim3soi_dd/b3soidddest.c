/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soidddest.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice.h"
#include "b3soidddef.h"
#include "suffix.h"

void
B3SOIDDdestroy(GENmodel **inModel)
{
B3SOIDDmodel **model = (B3SOIDDmodel**)inModel;
B3SOIDDinstance *here;
B3SOIDDinstance *prev = NULL;
B3SOIDDmodel *mod = *model;
B3SOIDDmodel *oldmod = NULL;

    for (; mod ; mod = mod->B3SOIDDnextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (B3SOIDDinstance *)NULL;
         for (here = mod->B3SOIDDinstances; here; here = here->B3SOIDDnextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



