/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifddest.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice.h"
#include "b3soifddef.h"
#include "suffix.h"

void
B3SOIFDdestroy(GENmodel **inModel)
{
B3SOIFDmodel **model = (B3SOIFDmodel**)inModel;
B3SOIFDinstance *here;
B3SOIFDinstance *prev = NULL;
B3SOIFDmodel *mod = *model;
B3SOIFDmodel *oldmod = NULL;

    for (; mod ; mod = mod->B3SOIFDnextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (B3SOIFDinstance *)NULL;
         for (here = mod->B3SOIFDinstances; here; here = here->B3SOIFDnextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



