/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipddest.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "b3soipddef.h"
#include "suffix.h"

void
B3SOIPDdestroy (inModel)
     GENmodel **inModel;
{
  B3SOIPDmodel **model = (B3SOIPDmodel **) inModel;
  B3SOIPDinstance *here;
  B3SOIPDinstance *prev = NULL;
  B3SOIPDmodel *mod = *model;
  B3SOIPDmodel *oldmod = NULL;

  for (; mod; mod = mod->B3SOIPDnextModel)
    {
      if (oldmod)
	FREE (oldmod);
      oldmod = mod;
      prev = (B3SOIPDinstance *) NULL;
      for (here = mod->B3SOIPDinstances; here;
	   here = here->B3SOIPDnextInstance)
	{
	  if (prev)
	    FREE (prev);
	  prev = here;
	}
      if (prev)
	FREE (prev);
    }
  if (oldmod)
    FREE (oldmod);
  *model = NULL;
  return;
}
