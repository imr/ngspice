/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdmdel.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "b3soipddef.h"
#include "sperror.h"
#include "suffix.h"

int
B3SOIPDmDelete (inModel, modname, kill)
     GENmodel **inModel;
     IFuid modname;
     GENmodel *kill;
{
  B3SOIPDmodel **model = (B3SOIPDmodel **) inModel;
  B3SOIPDmodel *modfast = (B3SOIPDmodel *) kill;
  B3SOIPDinstance *here;
  B3SOIPDinstance *prev = NULL;
  B3SOIPDmodel **oldmod;

  oldmod = model;
  for (; *model; model = &((*model)->B3SOIPDnextModel))
    {
      if ((*model)->B3SOIPDmodName == modname ||
	  (modfast && *model == modfast))
	goto delgot;
      oldmod = model;
    }
  return (E_NOMOD);

delgot:
  *oldmod = (*model)->B3SOIPDnextModel;	/* cut deleted device out of list */
  for (here = (*model)->B3SOIPDinstances; here;
       here = here->B3SOIPDnextInstance)
    {
      if (prev)
	FREE (prev);
      prev = here;
    }
  if (prev)
    FREE (prev);
  FREE (*model);
  return (OK);
}
