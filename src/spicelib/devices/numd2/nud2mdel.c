/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "numd2def.h"
#include "sperror.h"
#include "suffix.h"

int
NUMD2mDelete(inModel, modname, kill)
  GENmodel **inModel;
  IFuid modname;
  GENmodel *kill;

{
  NUMD2model **model = (NUMD2model **) inModel;
  NUMD2model *modfast = (NUMD2model *) kill;
  NUMD2instance *inst;
  NUMD2instance *prev = NULL;
  NUMD2model **oldmod;
  oldmod = model;
  for (; *model; model = &((*model)->NUMD2nextModel)) {
    if ((*model)->NUMD2modName == modname ||
	(modfast && *model == modfast))
      goto delgot;
    oldmod = model;
  }
  return (E_NOMOD);

delgot:
  *oldmod = (*model)->NUMD2nextModel;	/* cut deleted device out of list */
  for (inst = (*model)->NUMD2instances; inst; inst = inst->NUMD2nextInstance) {
    if (prev)
      FREE(prev);
    prev = inst;
  }
  if (prev)
    FREE(prev);
  FREE(*model);
  return (OK);

}
