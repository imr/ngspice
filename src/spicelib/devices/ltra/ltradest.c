/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice.h"
#include "ltradefs.h"
#include "suffix.h"

void
LTRAdestroy(GENmodel **inModel)
{
  LTRAmodel **model = (LTRAmodel **) inModel;
  LTRAinstance *here;
  LTRAinstance *prev = NULL;
  LTRAmodel *mod = *model;
  LTRAmodel *oldmod = NULL;

  for (; mod; mod = mod->LTRAnextModel) {
    if (oldmod)
      FREE(oldmod);
    oldmod = mod;
    prev = (LTRAinstance *) NULL;
    for (here = mod->LTRAinstances; here; here = here->LTRAnextInstance) {
      if (prev)
	FREE(prev);
      prev = here;
    }
    if (prev)
      FREE(prev);
  }
  if (oldmod)
    FREE(oldmod);
  *model = NULL;
}
