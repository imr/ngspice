/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice.h"
#include "ltradefs.h"
#include "sperror.h"
#include "suffix.h"

int
LTRAmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
  LTRAmodel **model = (LTRAmodel **) inModel;
  LTRAmodel *modfast = (LTRAmodel *) kill;
  LTRAinstance *here;
  LTRAinstance *prev = NULL;
  LTRAmodel **oldmod;
  oldmod = model;
  for (; *model; model = &((*model)->LTRAnextModel)) {
    if ((*model)->LTRAmodName == modname ||
	(modfast && *model == modfast))
      goto delgot;
    oldmod = model;
  }
  return (E_NOMOD);

delgot:
  *oldmod = (*model)->LTRAnextModel;	/* cut deleted device out of list */
  for (here = (*model)->LTRAinstances; here; here = here->LTRAnextInstance) {
    if (prev)
      FREE(prev);
    prev = here;
  }
  if (prev)
    FREE(prev);
  FREE(*model);
  return (OK);

}
