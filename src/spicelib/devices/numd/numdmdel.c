/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "numddefs.h"
#include "sperror.h"
#include "suffix.h"

int
NUMDmDelete(inModel, modname, kill)
  GENmodel **inModel;
  IFuid modname;
  GENmodel *kill;
{

  NUMDmodel **model = (NUMDmodel **) inModel;
  NUMDmodel *modfast = (NUMDmodel *) kill;
  NUMDinstance *inst;
  NUMDinstance *prev = NULL;
  NUMDmodel **oldmod;
  oldmod = model;
  for (; *model; model = &((*model)->NUMDnextModel)) {
    if ((*model)->NUMDmodName == modname ||
	(modfast && *model == modfast))
      goto delgot;
    oldmod = model;
  }
  return (E_NOMOD);

delgot:
  *oldmod = (*model)->NUMDnextModel;	/* cut deleted device out of list */
  for (inst = (*model)->NUMDinstances; inst; inst = inst->NUMDnextInstance) {
    if (prev)
      FREE(prev);
    prev = inst;
  }
  if (prev)
    FREE(prev);
  FREE(*model);
  return (OK);
}
